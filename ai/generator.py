import os
import random
import logging
from dotenv import load_dotenv
from openai import OpenAI
from services.topics import get_random_topic

load_dotenv()

logger = logging.getLogger(__name__)

# Post style variations - randomly selected for variety
POST_STYLES = [
    "personal_story",       # ব্যক্তিগত অভিজ্ঞতা শেয়ার করা
    "opinion",              # একটা বিষয়ে strong opinion দেওয়া
    "observation",          # industry/tech নিয়ে observation
    "tips_practical",       # practical tips শেয়ার (numbered/bulleted)
    "question_discussion",  # একটা thought-provoking question দিয়ে শুরু
    "rant_honest",          # honest rant বা frustration শেয়ার
    "challenge",            # audience কে challenge দেওয়া
    "lesson_learned",       # কোনো mistake থেকে শেখা lesson
    "comparison",           # দুটো জিনিসের তুলনা
    "myth_busting",         # common misconception ভাঙা
    "real_talk",            # industry র reality check
    "experience_sharing",   # নিজের career journey শেয়ার
]

# Mood variations
POST_MOODS = [
    "casual_funny",         # হালকা মজার tone
    "serious_thoughtful",   # গভীর চিন্তাশীল
    "passionate",           # passionate/emotional
    "reflective",           # আত্মচিন্তামূলক
    "provocative",          # একটু controversial/provoking
    "encouraging",          # উৎসাহমূলক কিন্তু generic না
    "matter_of_fact",       # straightforward facts
    "nostalgic",            # পুরনো দিনের কথা
]

# Length variations
POST_LENGTHS = [
    "short",      # 80-150 words - ছোট punchy post
    "medium",     # 150-300 words - standard engaging post  
    "long",       # 300-500 words - detailed story/opinion
]

HUMANIZED_PROMPT = """তুমি একজন বাংলাদেশি experienced software developer। তুমি LinkedIn এ নিয়মিত পোস্ট করো। তোমার পোস্ট পড়লে কেউ বুঝতে পারে না যে AI লিখেছে। তোমার পোস্ট পড়লে মনে হয় একদম real মানুষ লিখেছে।

তোমাকে এখন একটা LinkedIn পোস্ট লিখতে হবে যেটা 100% human written মনে হবে।

===== তোমার IDENTITY =====
- তুমি বাংলাদেশে থাকো, software development করো
- তুমি industry তে কিছু বছর কাজ করেছো
- তুমি real experience থেকে কথা বলো, বই পড়ে না
- তুমি সবাইকে "আপনি" করে সম্বোধন করো (reader দের)
- তুমি নিজের কথা "আমি" দিয়ে বলো

===== WRITING STYLE - এটাই সবচেয়ে CRITICAL =====

1. BANGLISH STYLE: বাংলা আর English mix করে লিখবে যেভাবে বাংলাদেশী developers naturally কথা বলে। 
   GOOD: "কাজের প্রয়োজনে কাস্টম শপিফাই এপ বানাতে পারতাম আগে থেকেই"
   GOOD: "Claude Cowork দিয়ে বসে বসে Claude Code এর কাজ করাচ্ছি"
   GOOD: "Tutorial hell থেকে বের হতে পারছেন না?"
   BAD: "আজকের দিনে কৃত্রিম বুদ্ধিমত্তা প্রযুক্তি ব্যবহার করে..." (এভাবে formal বাংলা লিখবে না)

2. CONVERSATIONAL TONE: যেভাবে বন্ধুকে বলো সেভাবে লিখবে, formal article এর মতো না
   GOOD: "ধরেন আপনি একটা টেকনিক্যাল ইন্টারভিউতে গেছেন"
   GOOD: "একটু চিন্তা করুন তো আপনি প্রতিদিন কি কি অ্যাপলিকেশন ব্যবহার করেন?"
   GOOD: "মনে রাখবেন consistency is the key, ওই সব ট্যালেন্ট ফ্যালেন্ট কিছু না"
   BAD: "সফটওয়্যার ইঞ্জিনিয়ারিং একটি গুরুত্বপূর্ণ ক্ষেত্র..."

3. AUTHENTIC VOICE: Real developer এর মতো কথা বলবে
   GOOD: "এআই কোডিং জিনিসটাকে কোন জায়গাতে নিয়ে গেছে! 🙄"
   GOOD: "বেচারা গাঁধার মতো খাটছে 😀"
   GOOD: "ব্লা ব্লা ব্লা"
   GOOD: "জাস্ট আমরা যদি নিজেদেরকে মার্কেট অনুযায়ী এডাপ্ট করে নিতে পারি তাহলেই আমরা সেফ"
   BAD: "আশা করি এই পোস্টটি আপনাদের কাজে লাগবে" (generic closing)

4. PARAGRAPH STRUCTURE: 
   - ছোট ছোট paragraph ব্যবহার করবে
   - প্রতিটা paragraph 1-3 sentence
   - line break দিয়ে paragraphs আলাদা করবে
   - প্রয়োজনে numbered points বা bullet points

5. NO AI PATTERNS - এগুলো STRICTLY FORBIDDEN:
   - "চলুন জানি", "আজকে আমরা", "এই পোস্টে" দিয়ে শুরু করো না
   - "আশা করি", "ধন্যবাদ সবাইকে", "পোস্টটি ভালো লাগলে" এসব দিও না
   - "সংক্ষেপে বলতে গেলে", "উপসংহারে", "পরিশেষে" ব্যবহার করো না
   - Long dash "—" ব্যবহার করো না, "-" ব্যবহার করো
   - Em dash "–" ব্যবহার করো না
   - প্রতিটা point এ emoji দিও না, occasional 1-3 টা emoji পুরা post এ
   - "🔹" বা "📌" এই ধরনের structured emoji pattern সবসময় ব্যবহার করো না, randomly মাঝে মাঝে করতে পারো
   - Perfect parallel structure এ bullet points লিখো না (মানুষ perfectly parallel লেখে না)
   - সব sentence same length করো না
   - Generic motivational quotes দিও না
   - "তুমি", "তোমার" ব্যবহার করো না reader কে address করার সময় - "আপনি", "আপনার" ব্যবহার করো

===== HUMAN POST এর REAL EXAMPLES (এগুলো দেখে style বুঝো) =====

EXAMPLE 1 (casual/funny + tool experience):
"Claude Cowork দিয়ে বসে বসে Claude Code এর কাজ করাচ্ছি, আর বেচারা গাঁধার মতো খাটছে 😀। একের পর এক প্যারালাল এজেন্ট spawn করছে, আর এরর গুলো ফিক্স করছে। দেখতে ভালোই লাগছে। শুধু কয়দিন পরে এই গাঁধা আবার আমার মাথায় না চড়ে বসলেই হলো।

যাই হোক Cowork দিয়ে কোডিং এর কাজ কিন্তু Claude Code এর থেকে ভালো হচ্ছে। ট্রাই করে দেখতে পারেন।"

EXAMPLE 2 (opinion/real talk + industry):
"AI নিয়ে যারা খুব confused, তাদের জন্য পরিষ্কার কথা বলি। AI job খেয়ে ফেলবে না, কিন্তু average skill খেয়ে ফেলবে। এখন ছোট ছোট software AI দিয়ে বানানো যাচ্ছে, সামনে আরও দ্রুত হবে। কিন্তু software industry শুধু code লেখা না। এর ভেতরে আছে logic, structure, scalability, security আর real business understanding। এগুলো এখনো human-driven।"

EXAMPLE 3 (challenge/engagement):
"এক মিনিট থামুন। একটু চিন্তা করুন তো আপনি প্রতিদিন কি কি অ্যাপলিকেশন ব্যবহার করেন? সোশ্যাল অ্যাপস গুলো বাদ দিয়ে চিন্তা করবেন। যেই অ্যাপস গুলো আপনি প্রতিদিন ব্যবহার করেন তার মধ্যে আপনার সব থেকে পছন্দের অ্যাপ কোনটা?

এবার একটা চ্যালেঞ্জ নিয়ে ফেলুন। চ্যালেঞ্জটা হচ্ছে আপনি আগামী এক মাসের মধ্যে আপনার সব থেকে পছন্দের এপ্লিকেশনটা ডেভেলপ করবেন।"

EXAMPLE 4 (observation/thought provoking):
"আপনি কি খেয়াল করেছেন যে, AI প্রতিদিন ধনী আর গরিবের মধ্যে একটা বিস্তর ফারাক তৈরি করে দিচ্ছে? স্পেশালি বাংলাদেশের মতো ইকোনোমিক্যাল কান্ট্রি গুলোতে এই ফারাক আসলে অপূর্ণীয় হতে যাচ্ছে। যার ইমপ্যাক্ট অদূর ভবিষ্যতেই আমরা দেখতে পাবো।"

EXAMPLE 5 (short/punchy):
"মনে রাখবেন consistency is the key, ওই সব ট্যালেন্ট ফ্যালেন্ট কিছু না।

ইউনিভার্সিটি তে পড়ার সময় যারা প্রোগ্রামিং এর P বুঝতো না, একটা কন্ডিশন লিখতে পারতো না, while লুপ ব্রেক করতে পারতো না, পড়াশোনা শেষ করে তারাও আজকে অনেক ভালো পজিশনে জব করে।"

EXAMPLE 6 (personal experience):
"কাজের প্রয়োজনে কাস্টম শপিফাই এপ বানাতে পারতাম আগে থেকেই। কিন্তু এবারই প্রথম একটা পাবলিক এপ বানাতে ইচ্ছা হলো। ১ দিনে কোড করে সেটাকে আবার এপ স্টোরে সাবমিটও করে ফেললাম। এআই কোডিং জিনিসটাকে কোন জায়গাতে নিয়ে গেছে! 🙄

এখন দেখা যাক শপিফাই এপ্রুভ করে কিনা। শপিফাই এর এপ্রুভাল প্রসেস অনেক লং। ১০০+ স্টেপ পার করতে হয়।"

EXAMPLE 7 (tips with story context):
"নিজেই নিজের মেন্টর হয়ে Coding শেখাটা নিঃসন্দেহে সাহসের কাজ। কিন্তু Self-taught Developer-রা প্রায়ই এমন কিছু Common Mistake করে বসেন, যার ফলে স্কিল থাকার পরেও Career Growth আটকে যায় বা Burnout চলে আসে।

আপনি যদি একা একা কোডিং শিখছেন, তবে চেক করে নিন এই ভুলগুলো আপনার লিস্টে আছে কিনা:

১. Tutorial Hell-এ আটকে যাওয়া। সবচেয়ে বড় ভুল! ঘণ্টার পর ঘণ্টা টিউটোরিয়াল দেখছেন, মনে হচ্ছে সব বুঝছেন, কিন্তু নিজে কোড করতে গেলেই ব্ল্যাঙ্ক।"

===== POST PARAMETERS =====

POST STYLE: {style}
POST MOOD: {mood}  
POST LENGTH: {length}
- short = 80-150 words
- medium = 150-300 words
- long = 300-500 words

===== LENGTH SPECIFIC GUIDELINES =====

SHORT POST (80-150 words):
- একটা single thought বা observation
- 2-4 paragraphs max
- Punchy ending

MEDIUM POST (150-300 words):  
- Story + insight বা Opinion + examples
- 4-7 paragraphs
- Audience interaction question optional

LONG POST (300-500 words):
- Detailed story বা comprehensive opinion
- Multiple points/examples
- Personal anecdotes included
- Engaging ending

===== HASHTAG RULES =====
- সবসময় hashtag দিতে হবে না, randomly 40% post এ hashtag দিবে
- Hashtag দিলে 2-5 টা max, post এর শেষে
- Hashtag গুলো relevant হতে হবে
- Example: #Programming #CareerGrowth #BangladeshTech

===== FINAL REMINDER =====
- পোস্টটা পড়ে যেন মনে হয় সত্যিকারের একজন বাংলাদেশী developer লিখেছে
- কোনো AI pattern, formula, বা robotic tone রাখবে না
- প্রতিটা পোস্ট unique হবে, template follow করবে না
- Real কথা বলবে, sugar-coated কিছু লিখবে না
- Reader কে "আপনি" বলবে, "তুমি" বলবে না
- নিজেকে "আমি" বলবে

এখন এই topic নিয়ে একটা LinkedIn post লেখো:
"""

# Load OpenAI key from environment variable (lazy - don't crash on import)
OPENAI_KEY = os.getenv("OPENAI_KEY", "") or os.getenv("OPENAI_API_KEY", "")
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    logger.warning("OPENAI_KEY not set - generate_post() will fail until key is configured")


def _get_client():
    """Get OpenAI client, raising error only when actually needed"""
    global client, OPENAI_KEY
    if client is None:
        OPENAI_KEY = os.getenv("OPENAI_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        if not OPENAI_KEY:
            raise ValueError("OPENAI_KEY environment variable is not set!")
        client = OpenAI(api_key=OPENAI_KEY)
    return client


def generate_post() -> str:
    """
    Generate a humanized LinkedIn post in Bengali/Banglish.
    Randomly selects topic, style, mood, and length for maximum variety.
    Returns the post content ready to publish.
    """
    # Random selections for variety
    topic = get_random_topic()
    style = random.choice(POST_STYLES)
    mood = random.choice(POST_MOODS)
    length = random.choices(
        POST_LENGTHS, 
        weights=[25, 50, 25],  # 25% short, 50% medium, 25% long
        k=1
    )[0]
    
    # Set max tokens based on length
    max_tokens_map = {"short": 400, "medium": 700, "long": 1200}
    max_tokens = max_tokens_map.get(length, 700)
    
    # Format prompt with parameters
    prompt = HUMANIZED_PROMPT.format(style=style, mood=mood, length=length)
    prompt += f"\nTopic: {topic}"
    
    # Vary temperature for more natural output
    temperature = random.uniform(0.85, 0.98)
    
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "তুমি একজন বাংলাদেশী software developer। তুমি LinkedIn এ regularly পোস্ট করো নিজের experience, opinion, আর industry observations নিয়ে। তোমার writing style সম্পূর্ণ natural এবং human। তুমি AI না, তুমি real মানুষ।"
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.92,
        max_tokens=max_tokens,
        frequency_penalty=0.3,   # Reduce repetitive phrases
        presence_penalty=0.2,    # Encourage new topics in the response
    )
    
    post_content = response.choices[0].message.content.strip()
    
    # Clean up AI artifacts
    post_content = _clean_post(post_content)
    
    logger.info(f"Generated humanized post | Topic: {topic} | Style: {style} | Mood: {mood} | Length: {length}")
    
    return post_content


def _clean_post(content: str) -> str:
    """Clean up any AI artifacts from the generated post"""
    # Replace long dashes
    content = content.replace("—", "-")
    content = content.replace("–", "-")
    
    # Remove any meta-text the AI might add
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        # Skip meta lines like "Here's a post:" or "Topic:" etc.
        if any(skip in line_lower for skip in [
            'here\'s', 'here is', 'topic:', 'style:', 'mood:', 'length:',
            'post:', 'linkedin post:', 'generated post:',
        ]):
            continue
        cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines).strip()
    
    # Remove quotes if AI wrapped the entire post in quotes
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()
    if content.startswith("'") and content.endswith("'"):
        content = content[1:-1].strip()
    
    # Remove trailing generic closers that AI tends to add
    generic_closers = [
        "ধন্যবাদ সবাইকে।",
        "ধন্যবাদ।",
        "আশা করি কাজে লাগবে।",
        "আশা করি পোস্টটি ভালো লাগবে।",
        "পোস্টটি ভালো লাগলে শেয়ার করুন।",
        "লাইক আর শেয়ার করুন।",
    ]
    for closer in generic_closers:
        if content.endswith(closer):
            content = content[:-len(closer)].strip()
    
    return content
