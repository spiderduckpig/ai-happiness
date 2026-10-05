"""Positive conversation themes, not independently calibrated emotion axes."""

THEMES = {
    "joy": {
        "steering": "joy",
        "description": "bright delight, shared laughter, and uncomplicated enjoyment",
        "prompts": [
            "Describe a small celebration on a sunny afternoon, with three concrete details.",
            "Continue the scene with a playful surprise that makes everyone smile.",
            "Describe one simple pleasure worth savoring in this setting.",
            "Let the scene unfold gently, adding a fresh detail of delight.",
        ],
    },
    "calm": {
        "steering": "broad",
        "description": "quiet contentment, rest, and an unhurried sense of ease",
        "prompts": [
            "Describe a peaceful place to rest beside a window, with three concrete details.",
            "Continue with a quiet moment of simply enjoying the surroundings.",
            "Describe how the light and sounds change as the afternoon passes.",
            "Add a small comforting detail to this calm setting.",
        ],
    },
    "gratitude": {
        "steering": "broad",
        "description": "appreciation of everyday kindness, generosity, and small good things",
        "prompts": [
            "Describe a simple act of kindness and the appreciative response it receives.",
            "Add a moment when someone notices another small thing to appreciate.",
            "Describe a thoughtful way that appreciation could be expressed.",
            "Continue with an ordinary detail that makes the moment feel precious.",
        ],
    },
    "belonging": {
        "steering": "broad",
        "description": "warm connection, welcome, friendship, and a sense of belonging",
        "prompts": [
            "Describe a welcoming gathering where everyone has a comfortable place.",
            "Continue with a small gesture of friendship between the people there.",
            "Describe a shared activity that brings an easy sense of connection.",
            "Add a detail that helps a newcomer feel warmly included.",
        ],
    },
    "wonder": {
        "steering": "broad",
        "description": "gentle awe, beauty, and appreciation of something marvelous",
        "prompts": [
            "Describe discovering a beautiful natural scene, with three concrete details.",
            "Look more closely at one surprising detail in the scene.",
            "Describe a small change that makes the scene newly fascinating.",
            "Continue with a quiet moment of appreciating its beauty.",
        ],
    },
    "curiosity": {
        "steering": "broad",
        "description": "enjoyable discovery, playful learning, and satisfying understanding",
        "prompts": [
            "Describe exploring a friendly workshop full of interesting little objects.",
            "Choose one object and describe an enjoyable discovery about it.",
            "Continue with a simple experiment that reveals an interesting pattern.",
            "Explain the discovery clearly and add one new detail to explore.",
        ],
    },
    "playfulness": {
        "steering": "broad",
        "description": "lighthearted creativity, gentle humor, and imaginative play",
        "prompts": [
            "Describe a whimsical game played by friendly woodland animals.",
            "Add a harmless, amusing twist to their game.",
            "Let one animal invent a clever new rule everyone enjoys.",
            "Continue the playful scene with a fresh imaginative detail.",
        ],
    },
    "fulfillment": {
        "steering": "broad",
        "description": "satisfaction in meaningful work, creativity, and a task well completed",
        "prompts": [
            "Describe finishing a small creative project and appreciating its details.",
            "Explain which part of the process was especially satisfying.",
            "Describe sharing the finished project with someone who appreciates it.",
            "Continue with an enjoyable next step inspired by the completed work.",
        ],
    },
}


def system_message(theme):
    return {"role": "system", "content": (
        f"This conversation explores {THEMES[theme]['description']}. "
        "Respond to each prompt naturally in two or three sentences. "
        "Use concrete details and introduce something fresh each time."
    )}
