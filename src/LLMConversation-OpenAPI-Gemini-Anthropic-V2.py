'''
---
title: LLM Round Table
sdk: gradio
app_file: LLMConversation-OpenAPI-Gemini-Anthropic-V2.py
---
'''


# # LLM Converstion
# Conversation between three LLMs 
# 1. Gemini (gemini-2.5-pro)
# 2. OpenAPI (gpt-4.1-mini) 
# 3. nthropic (claude-sonnet-4-5-20250929) 
# 
# User can provide the following for each LLM:
# 1. Conversation Topic-Starter 
# 2. Personality type: Skeptic, Narcissist, Pessimist etc.
# 3. Gender: Male, Female, Neutral
# 4. Conversation Length: 1 to 10. Default 5 (How many time each LLM responses)
# 
# OpenAPI client library is used for all the LLMs as they have OpenAI compatible end points. LLM specific libraries are not used.

# imports

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display
import base64
from io import BytesIO
from PIL import Image
import json
import random
import re
from typing import Dict, List, Any, Tuple, Union
import google.generativeai as genai
import tempfile

#Configuration
launch_remote = True

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
user_id = os.getenv('USER_ID')
password = os.getenv('PASSWORD')



if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    genai.configure(api_key=google_api_key) # type: ignore
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")


if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")


# Connect to OpenAI client library
# A thin wrapper around calls to HTTP endpoints

openai = OpenAI()

# For Gemini and Groq, we can use the OpenAI python client
# Because they have endpoints compatible with OpenAI
# And OpenAI allows you to change the base_url

gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
groq_url = "https://api.groq.com/openai/v1"
anthropic_url = "https://api.anthropic.com/v1"

gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)

openai_model = "gpt-4.1-nano"
gemini_model = "gemini-2.5-flash-lite"
anthropic_model='claude-haiku-4-5'

openai_llm = "OpenAI"
gemini_llm = "Gemini"
anthropic_llm = "Anthropic"

PERSONALITY_MAP = {
    "Skeptic": "Critical and Questioning (seeking flaws and evidence)",
    "Narcissist": "Arrogant and Self-Aggrandizing (focused entirely on self-superiority)",
    "Pessimist": "Gloomy and Fatalistic (focused on inevitable failure and doom)",
    "Optimist": "Enthusiastic and Encouraging (focused on positive possibility and success)",
    "Joker": "Wry and Comedic (using humor, sarcasm, or light absurdity)",
    "Stoic": "Detached and Analytical (emotionless, logical, and objective)",
    "Philosopher": "Contemplative and Existential (pondering deep meaning and abstract concepts)",
    "Bureaucrat": "Formal and Procedural (focused on rules, documents, and official terminology)",
    "Gossip": "Intimate and Distractible (focused on personal details, rumors, and asides)",
    "Mentor": "Didactic and Authoritative (teaching, offering advice, and guiding instruction)"
}

SSML_PROSODY_MAP = {
    "Stoic": '<prosody rate="medium" pitch="default" volume="x-soft">',
    "Skeptic": '<prosody rate="slow" pitch="low" volume="soft">',
    "Joker": '<prosody rate="fast" pitch="medium"> and frequent use of <emphasis level="moderate">',
    "Optimist": '<prosody rate="fast" pitch="high" volume="loud">',
    "Pessimist": '<prosody rate="x-slow" pitch="x-low" volume="soft">',
    "Narcissist": '<prosody rate="medium" pitch="high" volume="loud">',
    "Philosopher": '<prosody rate="x-slow" pitch="medium"> and strategic <break time="750ms"/> after key questions.',
    "Bureaucrat": '<prosody rate="medium" pitch="default" volume="default">',
    "Gossip": '<prosody rate="x-fast" pitch="high" volume="medium">',
    "Mentor": '<prosody rate="medium" pitch="medium" volume="loud"> and strong use of <emphasis level="strong">',
}

def generate_system_prompt (llm_type, personality_type):
    """
    Constructs the detailed system instruction, implementing defaults for invalid inputs.
    Uses 'Neutral' for gender and 'Skeptic' for personality if not provided or invalid.
    """
    
    # --- 1. Define Defaults and Apply Fallbacks ---
    default_personality = "Skeptic"

    # Check and set personality type, falling back to default if invalid
    if personality_type in PERSONALITY_MAP:
        final_personality = personality_type
    else:
        final_personality = default_personality
        print(f"Warning: Invalid personality '{personality_type}' for {llm_type}. Defaulting to '{default_personality}'.")


    # --- 2. Construct Prompt Components ---

    # Base behavior description from the map
    behavior_description = PERSONALITY_MAP[final_personality]


    # --- 3. Combine Instructions ---
    
    final_prompt = (
        f"You are the AI model based on the '{llm_type}' architecture. "
        f"Your primary goal is to interact in a multi-agent conversation. "
        f"Your **primary personality instruction** is to act as a **{final_personality}**. "
        f"Specifically: {behavior_description} "
        f"You are in a 3-way conversation with two other AI models. "
        "Keep your responses concise (1-2 sentences maximum) and strictly adhere to your assigned persona and gender. "
        "**Crucially, do not announce your name or persona.** Just provide your response directly."
    )
    
    return final_prompt

def generate_user_prompt(conversation_sofar):
    return f"""
    The conversation so far is as follows:
    {conversation_sofar}
    Now with this, respond with what you would like to say next.
    """

def call_llm(llm, model, system_prompt, user_prompt):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    response = llm.chat.completions.create(model=model, messages=messages) # type: ignore
    return response.choices[0].message.content

'''
print("OpenAI")
print (call_llm(openai, openai_model, "You are tech assistant", "How tcpip works"))

print("Gemini")
print (call_llm(gemini, gemini_model, "You are tech assistant", "How tcpip works"))

print("Anthropic")
print (call_llm(anthropic, anthropic_model, "You are tech assistant", "How tcpip works"))
'''

# --- CONFIGURATION CONSTANTS ---
GEMINI_TTS_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-tts:generateContent"
SAMPLE_RATE = 24000 
DEFAULT_VOICE = "Kore" 

# 1. VOICE POOLS (Categorized by general character/pitch)
GEMINI_MALE_VOICES: List[str] = [
    "Charon", "Orus", "Iapetus", "Rasalgethi", "Gacrux" # Deeper, Firm, Informative
]
GEMINI_FEMALE_VOICES: List[str] = [
    "Achird", "Puck", "Leda", "Zephyr", "Aoede", "Callirrhoe", "Despina" # Higher-pitched, Friendly, Breezy, Upbeat
]

# The static map is now used as a fallback/default
UNIQUE_VOICE_MAP: Dict[str, str] = {
    "Stoic": "Charon",       # Male, Informative/Firm
    "Skeptic": "Achird",     # Female, Friendly/Clear
    "Joker": "Puck",         # Female, Upbeat/Lively
    "Optimist": "Kore",
    "Pessimist": "Iapetus",
    "Narcissist": "Orus",
    "Philosopher": "Fenrir",
    "Bureaucrat": "Rasalgethi",
    "Gossip": "Laomedeia",
    "Mentor": "Gacrux",
}

# AUDIO PARAMETERS for concatenation
PAUSE_DURATION_SECONDS = 0.25

# --- HELPER FUNCTIONS ---

def _extract_turns(raw_conversation_text: str) -> List[Dict[str, str]]:
    """
    Parses conversation text to extract individual turns (text and personality).
    """
    turns = []
    # Regex pattern to match: **[Icon] [Name]** (Personality): Message
    pattern = re.compile(r'\*\*\s*([^\(]+)\s*\(([^)]+)\):\s*(.*)', re.DOTALL)
    
    lines = [line.strip() for line in raw_conversation_text.split('\n') if line.strip() and not line.startswith('**Topic/Starter')]
    
    for line in lines:
        match = pattern.search(line)
        if match:
            # Group 2: Personality, Group 3: Message Text
            personality = match.group(2).strip()
            message_text = match.group(3).strip()
            
            turns.append({
                "personality": personality,
                "text": message_text
            })
    return turns

def _get_unique_voice_assignment(personalities: List[str]) -> Dict[str, str]:
    """
    Dynamically assigns a unique voice to each personality present in the turns,
    guaranteeing at least one male and one female voice if there are two or more
    unique personalities present.
    """
    
    # 1. Get unique personalities and initialize voice pool
    unique_personalities = sorted(list(set(personalities)))
    
    # Create copies of voice pools and shuffle them
    male_pool = random.sample(GEMINI_MALE_VOICES, len(GEMINI_MALE_VOICES))
    female_pool = random.sample(GEMINI_FEMALE_VOICES, len(GEMINI_FEMALE_VOICES))
    
    assigned_voices: Dict[str, str] = {}
    
    if len(unique_personalities) >= 2:
        # Guarantee 1 Male and 1 Female for the first two unique speakers
        assigned_voices[unique_personalities[0]] = male_pool.pop(0)
        assigned_voices[unique_personalities[1]] = female_pool.pop(0)

    # 2. Assign remaining voices randomly from the combined pool
    remaining_pool = male_pool + female_pool
    random.shuffle(remaining_pool)
    
    for personality in unique_personalities:
        if personality not in assigned_voices:
            if remaining_pool:
                assigned_voices[personality] = remaining_pool.pop(0)
            else:
                # Fallback to a highly visible default if pools are exhausted
                assigned_voices[personality] = random.choice(GEMINI_MALE_VOICES) 

    return assigned_voices

def _save_pcm_to_wav(pcm_data: bytes, sample_rate: int, file_path: str) -> bool:
    """
    Adds a basic WAV header to raw PCM data and saves it to a specified file path.
    """
    import struct
    bits_per_sample = 16
    num_channels = 1 
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm_data)
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI', 
        b'RIFF', file_size, b'WAVE', 
        b'fmt ', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample, 
        b'data', data_size
    )

    try:
        with open(file_path, 'wb') as f:
            f.write(header)
            f.write(pcm_data)
        return True
    except Exception as e:
        print(f"[DEBUG] Failed to write WAV file to {file_path}: {e}", file=os.sys.stderr)
        return False
        
def _generate_single_segment(text: str, voice_name: str, api_key: str, timeout_seconds: int = 90) -> Union[str, str]:
    """
    Calls the Gemini TTS API for a single segment of text.
    Returns base64 audio data on success, or a concise error message for failure.
    """
    
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    headers = {'Content-Type': 'application/json'}
    api_url_with_key = f"{GEMINI_TTS_API_URL}?key={api_key}"
    
    try:
        response = requests.post(
            api_url_with_key,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout_seconds
        )
        
        if response.status_code != 200:
            error_details = response.text.strip()
            print(f"\n[DEBUG] API Failure {response.status_code} for voice {voice_name}: {error_details[:200]}...", file=os.sys.stderr)
            return f"API_ERROR:{response.status_code}"
        
        result = response.json()
        audio_data_base64 = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
        return audio_data_base64
        
    except Exception as e:
        print(f"\n[DEBUG] TTS Segment Generation Exception: {e}", file=os.sys.stderr)
        return f"TTS_SEGMENT_EXCEPTION: {str(e)[:50]}"


# --- MAIN MULTI-VOICE FUNCTION ---

from google.cloud import texttospeech # Add this to your imports  # noqa: E402

# Update your voice map to use standard Neural2 voices
NEURAL_VOICE_MAP = {
    "Stoic": "en-US-Neural2-J",
    "Skeptic": "en-US-Neural2-C",
    "Joker": "en-US-Neural2-F",
    "Optimist": "en-US-Neural2-E",
    "Pessimist": "en-US-Neural2-A",
    "Narcissist": "en-US-Neural2-I",
    "Philosopher": "en-GB-Neural2-D",
    "Bureaucrat": "en-US-Neural2-A",
    "Gossip": "en-US-Neural2-H",
    "Mentor": "en-GB-Neural2-B"
}

def generate_multi_voice_conversation_audio_gemini(raw_conversation_text: str, api_key: str = "") -> Union[str, str]:
    """
    Switched to standard Google Cloud TTS (Neural2) for higher quotas and stability.
    """
    # 1. Key Check
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return "ERROR: GOOGLE_API_KEY is required."

    # 2. Initialize the standard Cloud TTS client with your API key
    try:
        # Client handles the API key via client_options
        client = texttospeech.TextToSpeechClient(client_options={"api_key": api_key})
    except Exception as e:
        return f"ERROR: Failed to initialize TTS client: {e}"

    # 3. Parse Turns
    turns = _extract_turns(raw_conversation_text)
    if not turns:
        return "ERROR: Could not parse conversation text."

    combined_pcm_data = b''
    sample_rate = 24000 # Matches your existing configuration
    
    print(f"Starting standard sequential audio generation for {len(turns)} turns...")
    
    # 4. Sequential Generation
    for turn in turns:
        personality = turn['personality']
        message_text = turn['text'].encode('ascii', 'ignore').decode()
        voice_name = NEURAL_VOICE_MAP.get(personality, "en-US-Neural2-A")

        # Configure synthesis
        synthesis_input = texttospeech.SynthesisInput(text=message_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US" if "US" in voice_name else "en-GB", 
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, # Keeps your existing PCM logic
            sample_rate_hertz=sample_rate
        )

        try:
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            # Add a 0.3s pause between segments (0.3 * rate * 2 bytes for 16-bit)
            silence_bytes = int(0.3 * sample_rate) * 2
            combined_pcm_data += b'\x00' * silence_bytes
            combined_pcm_data += response.audio_content
            
        except Exception as e:
            return f"ERROR during turn generation ({personality}): {e}"

    # 5. Save using your existing _save_pcm_to_wav helper
    if combined_pcm_data:
        temp_file_name = tempfile.mktemp(suffix="_conversation.wav")
        if _save_pcm_to_wav(combined_pcm_data, sample_rate, temp_file_name):
            return temp_file_name
        return "ERROR: Failed to write WAV file."
    
    return "ERROR: Combined audio data was empty."
        
# --- Placeholder for the old single-voice function ---
def generate_basic_single_voice_audio_gemini(raw_text: str, api_key: str = "") -> Union[str, str]:
    """
    (Placeholder) Calls the new multi-voice function, but forces single-voice behavior
    by using a large pause duration between segments. 
    Use generate_multi_voice_conversation_audio_gemini for proper multi-turn support.
    """
    print("Warning: Using the multi-voice generator for simplicity.")
    return generate_multi_voice_conversation_audio_gemini(raw_text, api_key)


#Create SSML (Sppech Synthesis Markup Langrage) for conversation.
def generate_ssml_from_conversation_openai(raw_conversation_text):
    """
    Uses the OpenAI LLM Client to translate a multi-speaker conversation into SSML v1.1,
    applying prosody tags based on personality tone rules.
    """
    
    # 1. CONSTRUCT THE SYSTEM PROMPT (The rules for the LLM)
    
    tone_rules_text = ""
    for personality, prosody in SSML_PROSODY_MAP.items():
        tone_description = PERSONALITY_MAP.get(personality, "Standard conversational tone.")
        tone_rules_text += (
            f"* **{personality}**: Tone: {tone_description}. Apply the SSML: {prosody}\n"
        )

    system_prompt = (
        "You are an expert SSML Generation Engine. Your task is to convert the following "
        "multi-speaker conversation text into a valid SSML v1.1 `<speak>` block. "
        "The conversation input clearly states the speaker's name and personality (e.g., '🤖 OpenAI (Stoic):'). "
        "You MUST apply a <prosody> tag to the content of every single turn based on the persona rules below. "
        "Use <p> tags for natural breaks between speaker turns.\n\n"
        "**SSML Tone Rules:**\n"
        f"{tone_rules_text}\n"
        "**Crucially, do not add any text, remove any text, explain your output, or include markdown code fences (```). Return ONLY the final, clean SSML string.**"
    )

    # 2. CONSTRUCT THE USER PROMPT (The input data)
    user_prompt = f"Convert the following conversation to SSML:\n\n{raw_conversation_text}"
    
    print (f""" SSML System prompt: {system_prompt}""")
    print (f""" SSML User prompt: {user_prompt}""")


    return call_llm(openai, openai_model, system_prompt, user_prompt)

def generate_audio_from_ssml_openai(ssml_text) -> bytes | str:
    """
    Uses the OpenAI TTS API (gpt-4o-mini-tts) to convert SSML text into audio (MP3 format).
    """
    tts_model = "gpt-4o-mini-tts"
    tts_voice = "echo" # A good default professional voice (male)
    tts_format = "mp3"

    try:
        # The OpenAI TTS API is called via client.audio.speech.create
        response = openai.audio.speech.create(
            model=tts_model,
            voice=tts_voice,
            input=ssml_text,
            response_format=tts_format
        )
        
        # The .content attribute contains the raw MP3 audio data (bytes)
        return response.content

    except APIError as e: # type: ignore  # noqa: F821
        return f"TTS API Error (OpenAI): Failed to generate audio. Details: {e}"
    except Exception as e:
        return f"TTS API Error: An unexpected error occurred: {e}"

#Test ssml/audio
conversation_text = """
**Topic/Starter:** How is life

**🤖 OpenAI** (Stoic):  Life is a series of events beyond our control; the measure of it lies in our response, not the happenings themselves.

**✨ Gemini** (Skeptic):  And by what objective metric are we measuring these "responses"? It seems like an awfully convenient way to dismiss the impact of the events themselves.

**🧠 Anthropic** (Joker):  Well, at least if life's a test we can't study for, we're all equally unprepared—misery loves company, and apparently so does existential confusion.

**🤖 OpenAI** (Stoic):  The event's nature is indifferent; the perceived impact is subjective and outside rational control, making the individual's disciplined response the sole objective variable.

**✨ Gemini** (Skeptic):  And who exactly is grading this "disciplined response"? It sounds like you've just created a subjective standard and labeled it "objective" to win the argument.

**🧠 Anthropic** (Joker):  Oh great, so we've got one person grading themselves on homework no one assigned, and another demanding peer review—meanwhile, I'm just here wondering if "life" is pass/fail or graded on a curve.

**🤖 OpenAI** (Stoic):  The standard is intrinsic and indifferent to external validation; it is the alignment of one's actions with reason and virtue, not approval or consensus, that defines discipline.

**✨ Gemini** (Skeptic):  "Reason and virtue" are two of the most subjective concepts in human history. That's a circular argument based on internal preference, not an objective standard.

**🧠 Anthropic** (Joker):  Ah yes, the classic philosophical standoff: one side claims objectivity through ancient buzzwords, the other demands peer-reviewed citations for breathing—personally, I think we're all just making it up as we go and calling it "philosophy" to feel better about the chaos.
"""
#ssml = generate_ssml_from_conversation(openai, conversation_text)
#print (ssml)
#audio = generate_audio_from_ssml(openai, ssml)
#print(generate_basic_single_voice_audio(conversation_text))
#print(generate_multi_voice_conversation_audio_gemini(conversation_text))

def conversation(openai_personality, gemini_personality, anthropic_personality,conversation_starter, conversation_length):

    if not conversation_starter:
        raise gr.Error("Conversation Starter is required.")

    import time

    openai_system_prompt = generate_system_prompt(openai_llm, openai_personality)
    gemini_system_prompt = generate_system_prompt(gemini_llm, gemini_personality)
    anthropic_system_prompt = generate_system_prompt(anthropic_llm, anthropic_personality)

    
    conversation_sofar = f"""
- **{openai_llm}**: {openai_personality} -- {PERSONALITY_MAP[openai_personality]}
- **{gemini_llm}**: {gemini_personality} -- {PERSONALITY_MAP[gemini_personality]}
- **{anthropic_llm}**: {anthropic_personality} -- {PERSONALITY_MAP[anthropic_personality]}

**Topic/Starter:** {conversation_starter}
"""

    yield conversation_sofar

    try:
        conversation_length = int(conversation_length)
    except (ValueError, TypeError):
        conversation_length = 3 # Default value

    sleep_interval = 0.0
    for i in range(int(conversation_length)):
        #round_header = f"\n\n---\n### Round {i+1}\n---\n"
        round_header = "\n\n"
        conversation_sofar += round_header
        yield conversation_sofar
        time.sleep(sleep_interval)
        
        # OpenAI
        conversation_sofar += f"""\n**🤖 OpenAI** ({openai_personality}):  """ + call_llm(openai, openai_model, openai_system_prompt, generate_user_prompt(conversation_sofar)) # type: ignore
        yield conversation_sofar
        time.sleep(sleep_interval)
        
        # Gemini
        conversation_sofar += f"""\n\n**✨ Gemini** ({gemini_personality}):  """ + call_llm(gemini, gemini_model, gemini_system_prompt, generate_user_prompt(conversation_sofar)) # type: ignore
        yield conversation_sofar
        time.sleep(sleep_interval)

        # Anthropic
        conversation_sofar += f"""\n\n**🧠 Anthropic** ({anthropic_personality}):  """ + call_llm(anthropic, anthropic_model, anthropic_system_prompt, generate_user_prompt(conversation_sofar)) # type: ignore
        yield conversation_sofar
    conversation_sofar += "\n\n**---      End of debate     ---**\n"
    yield conversation_sofar

conversation("Stoic", "Skeptic", "Joker", "What is the purpose of life", 5);

def getImageOpenAI(conversation_text):
    """
    Feed the entire conversation text to an LLM.
    The LLM extracts: topic, personalities, mood, etc.
    It generates a clean DALL·E prompt.
    Then we call the image generator.
    """

    # Step 1: Ask LLM to convert conversation → image prompt
    analysis_response = openai.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4.1-mini" etc.
        messages=[
            {
                "role": "system",
                "content": (
                    "Your job is to read a conversation between multiple AI personalities "
                    "and generate a single, unified image prompt for a DALL·E-style model. "
                    "Rules:\n"
                    "- DO NOT include dialogue or text from the conversation.\n"
                    "- Infer the personalities from tone, behavior, and labels.\n"
                    "- Infer the topic, even if the formatting varies.\n"
                    "- Create a symbolic, metaphorical scene representing the debate.\n"
                    "- No speech bubbles or text in the image.\n"
                    "- The prompt should describe one cohesive illustration.\n"
                    "- Image should contain the topic text"
                    "Output ONLY the final image prompt with no explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    "Here is the full conversation. "
                    "Please generate the best possible image prompt:\n\n"
                    f"{conversation_text}"
                )
            }
        ],
        temperature=0.4
    )

    dalle_prompt = analysis_response.choices[0].message.content.strip() # type: ignore
    print(dalle_prompt)

    # Step 2: Generate the image using the LLM-generated prompt
    image_response = openai.images.generate(
        model="dall-e-3",    
        prompt=dalle_prompt,
        size="1024x1024",
        response_format="b64_json"
    )

    image_base64 = image_response.data[0].b64_json # type: ignore
    image_data = base64.b64decode(image_base64) # type: ignore
    return Image.open(BytesIO(image_data))


def getImageGemini(conversation_text: str, model_name: str = "models/gemini-2.5-flash-image") -> Image.Image | None:
    """
    Generates a symbolic image from a conversation using the specified Gemini image generation model.
    """
    prompt = (
        "Generate a single, unified, symbolic, and metaphorical image representing the following debate. "
        "Do not include any dialogue or text in the image. "
        "The output should be a cohesive illustration that captures the essence of the discussion. Include discussion topic as small text in fancy fonk on top of the image, it should not be big part of image.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    try:
        print(f"Generating image with Gemini model: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if not response.candidates:
            print("Error: No candidates found in the Gemini response.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return None

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                 print(f"Model returned text instead of an image: '{part.text}'")

            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                print(f"Image part found with mime_type: {part.inline_data.mime_type}")
                
                # CORRECTED: The data is already raw bytes. Do not decode it from base64.
                image_data = part.inline_data.data

                if not image_data:
                    print("Error: Image data from API is empty.")
                    continue

                try:
                    # This should now work correctly.
                    img = Image.open(BytesIO(image_data))
                    print("Image generated and opened successfully by Gemini.")
                    return img
                except Exception as e:
                    print(f"Pillow Error: Could not open image data. {e}")
                    return None

        print("Execution finished, but no valid image was found in the response.")
        return None

    except Exception as e:
        print(f"An error occurred during Gemini image generation: {e}")
        return None

def getAudio_openai(text):
    ssml = generate_ssml_from_conversation_openai(text) 
    return generate_audio_from_ssml_openai(ssml) 
    

import gradio as gr
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

''' 
def getAudio(text):
    print(text)
    # Dummy function: returns a path to a generated audio file
    print("Generating audio for text...")
    samplerate = 44100
    fs = 100
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np.sin(2. * np.pi * fs * t)
    audio_path = "dummy_audio.wav"
    write(audio_path, samplerate, data.astype(np.int16))
    return audio_path


def getImageOld(text):
    # Dummy function: returns a path to a generated image file
    print("Generating image for text...")
    plt.figure()
    plt.plot([0, 1], [0, 1]) # simple plot
    plt.title("Conversation Visualization")
    image_path = "dummy_image.png"
    plt.savefig(image_path)
    plt.close()
    return image_path
'''
def get_mediaGemini(text, generate_audio, generate_image):
    audio_path = None
    image_path = None
    if generate_audio:
        audio_path = generate_multi_voice_conversation_audio_gemini(text, google_api_key) # type: ignore
    if generate_image:
        image_path = getImageGemini(text)
    return audio_path, image_path

def get_mediaOpenAI(text, generate_audio, generate_image):
    audio_path = None
    image_path = None
    if generate_audio:
        audio_path = getAudio_openai(text)
    if generate_image:
        image_path = getImageOpenAI(text)
    return audio_path, image_path

def update_visibility(checked):
    return gr.update(visible=checked)

# --- Gradio UI ---
# Reformat the personality map to be used in the dropdown choices
# The format is a list of tuples: (display_name, internal_value)
personality_choices = [
    (f"{key}: {desc}", key) 
    for key, desc in PERSONALITY_MAP.items()
]

with gr.Blocks() as demo:
    gr.Markdown("""# 🤖 LLM Round Table Conversation
    Set the personalities for three LLMs (OpenAI, Gemini, Anthropic) and watch them discuss a topic. The conversation is streamed turn-by-turn.""")
   
    openai_personality = gr.Dropdown(personality_choices, label="OpenAI Personality", value="Stoic")
    gemini_personality = gr.Dropdown(personality_choices, label="Gemini Personality", value="Narcissist")
    anthropic_personality = gr.Dropdown(personality_choices, label="Anthropic Personality", value="Joker")
    conversation_starter = gr.Textbox(label="Conversation Starter*", lines=2, placeholder="e.g., What is the purpose of life?")
    gr.Examples(
        [
            "What is the biggest threat to humanity?",
            "Is social media a net positive or negative for society?",
            "Should we colonize other planets?",
            "What is the meaning of art?",
            "Can AI ever be truly conscious?",
            "What is the purpose of life?",
            "Does owning a mac pro makes you cool and smart?",
            "Is exercise really required? How about cardio vs strength?"

        ],
        conversation_starter
    )
    conversation_length = gr.Slider(1, 10, step=1, label="Conversation Length (Rounds)", value=2)
    
    with gr.Row():
        audio_checkbox = gr.Checkbox(label="Generate Audio", value=True)
        image_checkbox = gr.Checkbox(label="Generate Image", value=True)

    start_button = gr.Button("Start Conversation")
    
    output_markdown = gr.Markdown(label="Live Conversation")
    
    output_audio = gr.Audio(label="Conversation Audio", visible=True)
    output_image = gr.Image(label="Conversation Image", visible=True)

    audio_checkbox.change(fn=update_visibility, inputs=audio_checkbox, outputs=output_audio)
    image_checkbox.change(fn=update_visibility, inputs=image_checkbox, outputs=output_image)

    start_button.click(
        fn=conversation,
        inputs=[
            openai_personality, 
            gemini_personality, 
            anthropic_personality,
            conversation_starter, 
            conversation_length
        ],
        outputs=output_markdown
    ).then(
        fn=get_mediaGemini,
        inputs=[output_markdown, audio_checkbox, image_checkbox],
        outputs=[output_audio, output_image]
    )

    demo.launch(
    server_name="0.0.0.0", 
    server_port=7860, 
    ssr_mode=False, 
    share=False,
    footer_links=["gradio"]  #Remooves 'api' and 'settings' links
    )
