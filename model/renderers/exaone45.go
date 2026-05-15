package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

type Exaone45Renderer struct{}

func (r *Exaone45Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	if len(tools) > 0 {
		sb.WriteString("<|tool_declare|>\n")
		if err := writeExaone45Tools(&sb, tools); err != nil {
			return "", err
		}
		sb.WriteString("<|endofturn|>\n")
	}

	for i, message := range messages {
		content := strings.TrimSpace(message.Content)
		if i == 0 && message.Role == "system" {
			sb.WriteString("<|system|>\n")
			sb.WriteString(content)
			sb.WriteString("<|endofturn|>\n")
			continue
		}

		switch message.Role {
		case "system":
			sb.WriteString("<|system|>\n")
			sb.WriteString(content)
			sb.WriteString("<|endofturn|>\n")
		case "user":
			sb.WriteString("<|user|>\n")
			sb.WriteString(content)
			sb.WriteString("<|endofturn|>\n")
		case "assistant":
			sb.WriteString("<|assistant|>\n")
			sb.WriteString("<think>\n\n</think>\n\n")
			content = stripExaone4Thinking(content)
			sb.WriteString(content)
			if len(message.ToolCalls) > 0 {
				if content != "" {
					sb.WriteByte('\n')
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
			sb.WriteString("<|endofturn|>\n")
		case "tool":
			if i > 0 && messages[i-1].Role == "tool" {
				sb.WriteByte('\n')
			} else {
				sb.WriteString("<|tool|>\n")
			}
			sb.WriteString("<tool_result>")
			sb.WriteString(content)
			sb.WriteString("</tool_result>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString("<|endofturn|>\n")
			}
		}
	}

	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		sb.WriteString("<|assistant|>\n")
		if think == nil || think.Bool() {
			sb.WriteString("<think>\n")
		} else {
			sb.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return sb.String(), nil
}

func writeExaone45Tools(sb *strings.Builder, tools []api.Tool) error {
	sb.WriteString("# Tools\n")
	for _, tool := range tools {
		toolJSON, err := exaone4JSON(tool)
		if err != nil {
			return err
		}
		sb.WriteString("<tool>")
		sb.WriteString(toolJSON)
		sb.WriteString("</tool>\n")
	}
	return nil
}
