package parsers

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

const (
	exaoneThinkingOpenTag  = "<think>"
	exaoneThinkingCloseTag = "</think>"
	exaoneToolCallOpenTag  = "<tool_call>"
	exaoneToolCallCloseTag = "</tool_call>"
)

type exaoneParserState int

const (
	exaoneParserStateThinking exaoneParserState = iota
	exaoneParserStateContent
	exaoneParserStateTool
)

type ExaoneParser struct {
	state                 exaoneParserState
	buffer                strings.Builder
	callIndex             int
	defaultThinking       bool
	allowLeadingThinkOpen bool
}

func (p *ExaoneParser) HasToolSupport() bool {
	return true
}

func (p *ExaoneParser) HasThinkingSupport() bool {
	return true
}

func (p *ExaoneParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buffer.Reset()
	p.callIndex = 0

	thinkingEnabled := p.defaultThinking
	if thinkValue != nil {
		thinkingEnabled = thinkValue.Bool()
	}

	assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""
	if thinkingEnabled && !assistantPrefill {
		p.state = exaoneParserStateThinking
		p.allowLeadingThinkOpen = true
	} else {
		p.state = exaoneParserStateContent
		p.allowLeadingThinkOpen = false
	}

	return tools
}

func (p *ExaoneParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	var contentSB, thinkingSB strings.Builder

	for {
		progress := false
		switch p.state {
		case exaoneParserStateThinking:
			var parsedThinking string
			progress, parsedThinking = p.consumeThinking(done)
			thinkingSB.WriteString(parsedThinking)
		case exaoneParserStateContent:
			var parsedContent string
			var parsedCalls []api.ToolCall
			progress, parsedContent, parsedCalls, err = p.consumeContent(done)
			if err != nil {
				return "", "", nil, err
			}
			contentSB.WriteString(parsedContent)
			calls = append(calls, parsedCalls...)
		case exaoneParserStateTool:
			var parsedCalls []api.ToolCall
			progress, parsedCalls, err = p.consumeTool(done)
			if err != nil {
				return "", "", nil, err
			}
			calls = append(calls, parsedCalls...)
		}
		if !progress {
			break
		}
	}

	return contentSB.String(), thinkingSB.String(), calls, nil
}

func (p *ExaoneParser) consumeThinking(done bool) (bool, string) {
	acc := p.buffer.String()
	if p.allowLeadingThinkOpen {
		trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
		if strings.HasPrefix(trimmed, exaoneThinkingOpenTag) {
			after := strings.TrimLeftFunc(strings.TrimPrefix(trimmed, exaoneThinkingOpenTag), unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.allowLeadingThinkOpen = false
			return true, ""
		}
		if strings.HasPrefix(exaoneThinkingOpenTag, trimmed) && !done {
			return false, ""
		}
		p.allowLeadingThinkOpen = false
	}

	if idx := strings.Index(acc, exaoneThinkingCloseTag); idx != -1 {
		thinking := acc[:idx]
		after := strings.TrimLeftFunc(acc[idx+len(exaoneThinkingCloseTag):], unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = exaoneParserStateContent
		return true, thinking
	}

	if idx := strings.Index(acc, exaoneToolCallOpenTag); idx != -1 {
		thinking := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		after := acc[idx+len(exaoneToolCallOpenTag):]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = exaoneParserStateTool
		return true, thinking
	}

	if done {
		p.buffer.Reset()
		p.state = exaoneParserStateContent
		return acc != "", acc
	}

	overlapLen := max(overlap(acc, exaoneThinkingCloseTag), overlap(acc, exaoneToolCallOpenTag))
	keep := trailingWhitespaceLen(acc)
	if overlapLen > 0 {
		beforePartialTag := acc[:len(acc)-overlapLen]
		keep = overlapLen + trailingWhitespaceLen(beforePartialTag)
	}
	if keep > 0 && keep < len(acc) {
		emit := acc[:len(acc)-keep]
		p.buffer.Reset()
		p.buffer.WriteString(acc[len(acc)-keep:])
		return emit != "", emit
	}
	return false, ""
}

func (p *ExaoneParser) consumeContent(done bool) (bool, string, []api.ToolCall, error) {
	acc := p.buffer.String()
	if idx := strings.Index(acc, exaoneToolCallOpenTag); idx != -1 {
		content := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		after := acc[idx+len(exaoneToolCallOpenTag):]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = exaoneParserStateTool
		return true, content, nil, nil
	}

	if done {
		p.buffer.Reset()
		return acc != "", acc, nil, nil
	}

	overlapLen := overlap(acc, exaoneToolCallOpenTag)
	keep := trailingWhitespaceLen(acc)
	if overlapLen > 0 {
		beforePartialTag := acc[:len(acc)-overlapLen]
		keep = overlapLen + trailingWhitespaceLen(beforePartialTag)
	}
	if keep > 0 && keep < len(acc) {
		emit := acc[:len(acc)-keep]
		p.buffer.Reset()
		p.buffer.WriteString(acc[len(acc)-keep:])
		return emit != "", emit, nil, nil
	}
	if keep == 0 && acc != "" {
		p.buffer.Reset()
		return true, acc, nil, nil
	}
	return false, "", nil, nil
}

func (p *ExaoneParser) consumeTool(done bool) (bool, []api.ToolCall, error) {
	acc := p.buffer.String()
	if idx := strings.Index(acc, exaoneToolCallCloseTag); idx != -1 {
		raw := acc[:idx]
		after := strings.TrimLeftFunc(acc[idx+len(exaoneToolCallCloseTag):], unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = exaoneParserStateContent
		calls, err := p.parseToolCalls(raw)
		return true, calls, err
	}
	if done && strings.TrimSpace(acc) != "" {
		p.buffer.Reset()
		p.state = exaoneParserStateContent
		calls, err := p.parseToolCalls(acc)
		return true, calls, err
	}
	return false, nil, nil
}

func (p *ExaoneParser) parseToolCalls(raw string) ([]api.ToolCall, error) {
	calls, err := parseExaoneToolCalls(raw)
	if err != nil {
		return nil, err
	}
	for i := range calls {
		calls[i].Function.Index = p.callIndex
		p.callIndex++
	}
	return calls, nil
}

func parseExaoneToolCalls(raw string) ([]api.ToolCall, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, fmt.Errorf("empty EXAONE tool call")
	}

	if strings.HasPrefix(raw, "[") {
		var parsed []exaoneHermesToolCall
		if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
			return nil, fmt.Errorf("failed to parse EXAONE JSON tool call array: %w", err)
		}
		calls := make([]api.ToolCall, 0, len(parsed))
		for _, call := range parsed {
			toolCall, err := call.toAPIToolCall()
			if err != nil {
				return nil, err
			}
			calls = append(calls, toolCall)
		}
		return calls, nil
	}

	var parsed exaoneHermesToolCall
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse EXAONE JSON tool call: %w", err)
	}
	toolCall, err := parsed.toAPIToolCall()
	if err != nil {
		return nil, err
	}
	return []api.ToolCall{toolCall}, nil
}

type exaoneHermesToolCall struct {
	Name      string                        `json:"name"`
	Arguments api.ToolCallFunctionArguments `json:"arguments"`
}

func (c exaoneHermesToolCall) toAPIToolCall() (api.ToolCall, error) {
	name := strings.TrimSpace(c.Name)
	if name == "" {
		return api.ToolCall{}, fmt.Errorf("empty EXAONE tool call name")
	}
	if c.Arguments.Len() == 0 {
		c.Arguments = api.NewToolCallFunctionArguments()
	}
	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: c.Arguments,
		},
	}, nil
}
