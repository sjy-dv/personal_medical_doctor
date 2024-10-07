package prompts

import "github.com/sjy-dv/personal_medical_doctor/langchain/llms"

var _ llms.PromptValue = StringPromptValue("")

// StringPromptValue is a prompt value that is a string.
type StringPromptValue string

func (v StringPromptValue) String() string {
	return string(v)
}

// Messages returns a single-element ChatMessage slice.
func (v StringPromptValue) Messages() []llms.ChatMessage {
	return []llms.ChatMessage{
		llms.HumanChatMessage{Content: string(v)},
	}
}
