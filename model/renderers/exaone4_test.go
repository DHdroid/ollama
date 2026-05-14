package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestExaone4RendererMatchesApplyChatTemplate(t *testing.T) {
	weatherTool := []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"location"},
				Properties: testPropsOrdered([]orderedProp{{
					Key: "location",
					Value: api.ToolProperty{
						Type:        api.PropertyType{"string"},
						Description: "City",
					},
				}}),
			},
		},
	}}

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{
			name:     "user_only_default",
			messages: []api.Message{{Role: "user", Content: "Reply with exactly: OK"}},
			want:     "[|user|]\nReply with exactly: OK[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name:     "user_only_enable_thinking_false",
			messages: []api.Message{{Role: "user", Content: "Reply with exactly: OK"}},
			think:    &api.ThinkValue{Value: false},
			want:     "[|user|]\nReply with exactly: OK[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name:     "user_only_enable_thinking_true",
			messages: []api.Message{{Role: "user", Content: "Reply with exactly: OK"}},
			think:    &api.ThinkValue{Value: true},
			want:     "[|user|]\nReply with exactly: OK[|endofturn|]\n[|assistant|]\n<think>\n",
		},
		{
			name: "system_user_default",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Hi"},
			},
			want: "[|system|]\nStay concise.[|endofturn|]\n[|user|]\nHi[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name: "multi_turn_default",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "Again"},
			},
			want: "[|user|]\nHi[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\nHello![|endofturn|]\n[|user|]\nAgain[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name: "assistant_with_thinking_stripped",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "<think>hidden</think>\n\nHello!"},
				{Role: "user", Content: "Again"},
			},
			want: "[|user|]\nHi[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\nHello![|endofturn|]\n[|user|]\nAgain[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name:     "tools_without_system",
			messages: []api.Message{{Role: "user", Content: "Weather?"}},
			tools:    weatherTool,
			want:     "[|system|]\n# Available Tools\nYou can use none, one, or multiple of the following tools by calling them as functions to help with the user’s query.\nHere are the tools available to you in JSON format within <tool> and </tool> tags:\n<tool>{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"location\"], \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}}}}}</tool>\n\nFor each function call you want to make, return a JSON object with function name and arguments within <tool_call> and </tool_call> tags, like:\n<tool_call>{\"name\": function_1_name, \"arguments\": {argument_1_name: argument_1_value, argument_2_name: argument_2_value}}</tool_call>\n<tool_call>{\"name\": function_2_name, \"arguments\": {...}}</tool_call>\n...\nNote that if no argument name is specified for a tool, you can just print the argument value directly, without the argument name or JSON formatting.[|endofturn|]\n[|user|]\nWeather?[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name: "tools_with_system",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
			},
			tools: weatherTool,
			want:  "[|system|]\nStay concise.\n\n# Available Tools\nYou can use none, one, or multiple of the following tools by calling them as functions to help with the user’s query.\nHere are the tools available to you in JSON format within <tool> and </tool> tags:\n<tool>{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"location\"], \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}}}}}</tool>\n\nFor each function call you want to make, return a JSON object with function name and arguments within <tool_call> and </tool_call> tags, like:\n<tool_call>{\"name\": function_1_name, \"arguments\": {argument_1_name: argument_1_value, argument_2_name: argument_2_value}}</tool_call>\n<tool_call>{\"name\": function_2_name, \"arguments\": {...}}</tool_call>\n...\nNote that if no argument name is specified for a tool, you can just print the argument value directly, without the argument name or JSON formatting.[|endofturn|]\n[|user|]\nWeather?[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
		{
			name: "assistant_tool_call_and_tool_result",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: testArgs(map[string]any{"location": "Seoul"}),
						},
					}},
				},
				{Role: "tool", Content: "Sunny"},
			},
			want: "[|user|]\nWeather?[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seoul\"}}</tool_call>[|endofturn|]\n[|tool|]\n<tool_result>{\"result\": \"Sunny\"}</tool_result>[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n",
		},
	}

	renderer := &Exaone4Renderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("rendered prompt mismatch\nwant: %q\n got: %q", tt.want, got)
			}
		})
	}
}
