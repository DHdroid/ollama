package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func boolThink(v bool) *api.ThinkValue {
	return &api.ThinkValue{Value: v}
}

func TestExaoneParserRegistry(t *testing.T) {
	tests := []string{"exaone4", "exaone4_5"}
	for _, name := range tests {
		parser := ParserForName(name)
		if parser == nil {
			t.Fatalf("%s parser = nil", name)
		}
		if _, ok := parser.(*ExaoneParser); !ok {
			t.Fatalf("%s parser = %T, want *ExaoneParser", name, parser)
		}
		if !parser.HasToolSupport() {
			t.Fatalf("%s parser should advertise tool support", name)
		}
		if !parser.HasThinkingSupport() {
			t.Fatalf("%s parser should advertise thinking support", name)
		}
	}
}

func TestExaone4DefaultThinkingOff(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add("reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "reason</think>answer" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaone4ThinkTrue(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, boolThink(true))

	content, thinking, calls, err := parser.Add("reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "reason" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaone45DefaultThinkingOn(t *testing.T) {
	parser := ParserForName("exaone4_5")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add("reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "reason" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaone45ThinkFalse(t *testing.T) {
	parser := ParserForName("exaone4_5")
	parser.Init(nil, nil, boolThink(false))

	content, thinking, calls, err := parser.Add("reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "reason</think>answer" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaoneLeadingThinkOpenRemovedOnce(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, boolThink(true))

	content, thinking, calls, err := parser.Add("<think>reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "reason" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaoneAssistantPrefillStartsInContentMode(t *testing.T) {
	parser := ParserForName("exaone4_5")
	parser.Init(nil, &api.Message{Role: "assistant", Content: "prefill"}, nil)

	content, thinking, calls, err := parser.Add("reason</think>answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "reason</think>answer" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
}

func TestExaoneToolCall(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add(`<tool_call>{"name":"get_weather","arguments":{"location":"Seoul"}}</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	requireExaoneToolCall(t, calls, 0, "get_weather", map[string]any{"location": "Seoul"})
}

func TestExaoneThinkingThenToolCall(t *testing.T) {
	parser := ParserForName("exaone4_5")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add(`reason</think>answer <tool_call>{"name":"get_weather","arguments":{"location":"Seoul"}}</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "reason" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	requireExaoneToolCall(t, calls, 0, "get_weather", map[string]any{"location": "Seoul"})
}

func TestExaoneThinkingPrematureToolCall(t *testing.T) {
	parser := ParserForName("exaone4_5")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add(`reason <tool_call>{"name":"get_weather","arguments":{"location":"Seoul"}}</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "reason" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	requireExaoneToolCall(t, calls, 0, "get_weather", map[string]any{"location": "Seoul"})
}

func TestExaoneStreamingSplitToolTag(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add("answer <too", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "" || len(calls) != 0 {
		t.Fatalf("first chunk content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add(`l_call>{"name":"get_weather","arguments":{"location":"Seoul"}}</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("second chunk content=%q thinking=%q", content, thinking)
	}
	requireExaoneToolCall(t, calls, 0, "get_weather", map[string]any{"location": "Seoul"})
}

func TestExaoneMultipleToolCalls(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, nil)

	_, _, calls, err := parser.Add(`<tool_call>{"name":"first","arguments":{"value":1}}</tool_call><tool_call>{"name":"second","arguments":{"value":2}}</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	requireExaoneToolCall(t, calls[:1], 0, "first", map[string]any{"value": float64(1)})
	requireExaoneToolCall(t, calls[1:], 1, "second", map[string]any{"value": float64(2)})
}

func TestExaoneHermesArrayToolCalls(t *testing.T) {
	parser := ParserForName("exaone4")
	parser.Init(nil, nil, nil)

	_, _, calls, err := parser.Add(`<tool_call>[{"name":"first","arguments":{"value":1}},{"name":"second","arguments":{"value":2}}]</tool_call>`, true)
	if err != nil {
		t.Fatal(err)
	}
	requireExaoneToolCall(t, calls[:1], 0, "first", map[string]any{"value": float64(1)})
	requireExaoneToolCall(t, calls[1:], 1, "second", map[string]any{"value": float64(2)})
}

func requireExaoneToolCall(t *testing.T, calls []api.ToolCall, index int, name string, wantArgs map[string]any) {
	t.Helper()
	if len(calls) != 1 {
		t.Fatalf("calls=%d, want 1", len(calls))
	}
	if calls[0].Function.Index != index {
		t.Fatalf("index=%d, want %d", calls[0].Function.Index, index)
	}
	if calls[0].Function.Name != name {
		t.Fatalf("name=%q, want %q", calls[0].Function.Name, name)
	}
	for key, want := range wantArgs {
		got, ok := calls[0].Function.Arguments.Get(key)
		if !ok {
			t.Fatalf("missing argument %q", key)
		}
		if got != want {
			t.Fatalf("%s=%v (%T), want %v (%T)", key, got, got, want, want)
		}
	}
}
