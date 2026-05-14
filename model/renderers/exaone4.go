package renderers

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type Exaone4Renderer struct{}

func (r *Exaone4Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	hasTools := len(tools) > 0
	for i, message := range messages {
		content := strings.TrimSpace(message.Content)
		if i == 0 {
			if message.Role == "system" {
				sb.WriteString("[|system|]\n")
				sb.WriteString(content)
				if hasTools {
					sb.WriteString("\n\n")
					if err := writeExaone4Tools(&sb, tools); err != nil {
						return "", err
					}
				}
				sb.WriteString("[|endofturn|]\n")
				continue
			}
			if hasTools {
				sb.WriteString("[|system|]\n")
				if err := writeExaone4Tools(&sb, tools); err != nil {
					return "", err
				}
				sb.WriteString("[|endofturn|]\n")
			}
		}

		switch message.Role {
		case "system":
			sb.WriteString("[|system|]\n")
			sb.WriteString(content)
			sb.WriteString("[|endofturn|]\n")
		case "user":
			sb.WriteString("[|user|]\n")
			sb.WriteString(content)
			sb.WriteString("[|endofturn|]\n")
		case "assistant":
			sb.WriteString("[|assistant|]\n")
			if content != "" {
				sb.WriteString("<think>\n\n</think>\n\n")
				sb.WriteString(stripExaone4Thinking(content))
			}
			if len(message.ToolCalls) > 0 {
				if content != "" {
					sb.WriteByte('\n')
				} else {
					sb.WriteString("<think>\n\n</think>\n\n")
				}
				for j, toolCall := range message.ToolCalls {
					if j > 0 {
						sb.WriteByte('\n')
					}
					args, err := exaone4JSON(toolCall.Function.Arguments)
					if err != nil {
						return "", err
					}
					sb.WriteString(`<tool_call>{"name": "`)
					sb.WriteString(toolCall.Function.Name)
					sb.WriteString(`", "arguments": `)
					sb.WriteString(args)
					sb.WriteString(`}</tool_call>`)
				}
			}
			sb.WriteString("[|endofturn|]\n")
		case "tool":
			if i > 0 && messages[i-1].Role == "tool" {
				sb.WriteByte('\n')
			} else {
				sb.WriteString("[|tool|]\n")
			}
			result, err := exaone4JSON(map[string]string{"result": content})
			if err != nil {
				return "", err
			}
			sb.WriteString("<tool_result>")
			sb.WriteString(result)
			sb.WriteString("</tool_result>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString("[|endofturn|]\n")
			}
		}
	}
	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		sb.WriteString("[|assistant|]\n")
		if think != nil && think.Bool() {
			sb.WriteString("<think>\n")
		} else {
			sb.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return sb.String(), nil
}

func writeExaone4Tools(sb *strings.Builder, tools []api.Tool) error {
	sb.WriteString("# Available Tools")
	sb.WriteString("\nYou can use none, one, or multiple of the following tools by calling them as functions to help with the user’s query.")
	sb.WriteString("\nHere are the tools available to you in JSON format within <tool> and </tool> tags:\n")
	for _, tool := range tools {
		toolJSON, err := exaone4JSON(tool)
		if err != nil {
			return err
		}
		sb.WriteString("<tool>")
		sb.WriteString(toolJSON)
		sb.WriteString("</tool>\n")
	}
	sb.WriteString("\nFor each function call you want to make, return a JSON object with function name and arguments within <tool_call> and </tool_call> tags, like:")
	sb.WriteString("\n<tool_call>{\"name\": function_1_name, \"arguments\": {argument_1_name: argument_1_value, argument_2_name: argument_2_value}}</tool_call>")
	sb.WriteString("\n<tool_call>{\"name\": function_2_name, \"arguments\": {...}}</tool_call>\n...")
	sb.WriteString("\nNote that if no argument name is specified for a tool, you can just print the argument value directly, without the argument name or JSON formatting.")
	return nil
}

func stripExaone4Thinking(content string) string {
	if _, after, ok := strings.Cut(content, "</think>"); ok {
		return strings.TrimSpace(after)
	}
	return content
}

func exaone4JSON(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	return exaone4PythonJSONSpacing(strings.TrimSuffix(buf.String(), "\n")), nil
}

func exaone4PythonJSONSpacing(s string) string {
	var sb strings.Builder
	sb.Grow(len(s) + strings.Count(s, ",") + strings.Count(s, ":"))
	inString := false
	escaped := false
	for _, r := range s {
		sb.WriteRune(r)
		if inString {
			if escaped {
				escaped = false
			} else if r == '\\' {
				escaped = true
			} else if r == '"' {
				inString = false
			}
			continue
		}
		switch r {
		case '"':
			inString = true
		case ',', ':':
			sb.WriteByte(' ')
		}
	}
	return sb.String()
}
