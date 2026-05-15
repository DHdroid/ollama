package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestExaone45RendererMatchesOfficialTemplate(t *testing.T) {
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
			name:     "user_only_default_thinking",
			messages: []api.Message{{Role: "user", Content: "Reply with exactly: OK"}},
			want:     "<|user|>\nReply with exactly: OK<|endofturn|>\n<|assistant|>\n<think>\n",
		},
		{
			name:     "user_only_thinking_false",
			messages: []api.Message{{Role: "user", Content: "Reply with exactly: OK"}},
			think:    &api.ThinkValue{Value: false},
			want:     "<|user|>\nReply with exactly: OK<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\n",
		},
		{
			name: "system_user_default",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Hi"},
			},
			want: "<|system|>\nStay concise.<|endofturn|>\n<|user|>\nHi<|endofturn|>\n<|assistant|>\n<think>\n",
		},
		{
			name:     "tools_without_system",
			messages: []api.Message{{Role: "user", Content: "Weather?"}},
			tools:    weatherTool,
			want:     "<|tool_declare|>\n# Tools\n<tool>{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"location\"], \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}}}}}</tool>\n<|endofturn|>\n<|user|>\nWeather?<|endofturn|>\n<|assistant|>\n<think>\n",
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
			want: "<|user|>\nWeather?<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seoul\"}}</tool_call><|endofturn|>\n<|tool|>\n<tool_result>Sunny</tool_result><|endofturn|>\n<|assistant|>\n<think>\n",
		},
	}

	renderer := &Exaone45Renderer{}
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

func TestExaoneRendererRegistryUsesSeparateRenderers(t *testing.T) {
	if _, ok := rendererForName("exaone4").(*Exaone4Renderer); !ok {
		t.Fatalf("exaone4 renderer = %T, want *Exaone4Renderer", rendererForName("exaone4"))
	}
	if _, ok := rendererForName("exaone4_5").(*Exaone45Renderer); !ok {
		t.Fatalf("exaone4_5 renderer = %T, want *Exaone45Renderer", rendererForName("exaone4_5"))
	}
}
