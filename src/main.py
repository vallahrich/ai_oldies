from speech_recognition.recognizer import SpeechRecognizer
from tts.synthesizer import TextToSpeech
from nlp.intent_classifier import IntentClassifier
from skills.reminder import ReminderSkill

class VoiceAssistant:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        self.intent_classifier = IntentClassifier()
        self.reminder_skill = ReminderSkill()

    def run(self):
        # Train the model if it's not already trained
        self.intent_classifier.train()
        # Or load a pre-trained model
        # self.intent_classifier.load_model('path/to/model.pth')

        self.tts.speak("Hello! I'm your voice assistant. How can I help you today?")
        while True:
            user_input = self.speech_recognizer.listen()
            if user_input:
                intent = self.intent_classifier.predict(user_input)
                self.handle_intent(intent, user_input)

    def handle_intent(self, intent, user_input):
        if intent == "greeting":
            self.tts.speak("Hello! How are you doing today?")
        elif intent == "set_reminder":
            # Simplified version, you'd need to extract task and time from user_input
            response = self.reminder_skill.set_reminder("take medicine", "14:00")
            self.tts.speak(response)
        elif intent == "goodbye":
            self.tts.speak("Goodbye! Have a great day!")
            exit()
        else:
            self.tts.speak("I'm sorry, I didn't understand that. Can you please repeat?")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
