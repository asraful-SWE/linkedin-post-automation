# LinkedIn Auto Poster

একটি intelligent automated LinkedIn posting system যা AI-powered Bengali content generate করে এবং natural behavior সহ posting করে।

## 🌟 Features

### Core Features
- ✅ **Bengali Content Generation**: AI দিয়ে authentic Bengali LinkedIn posts তৈরি
- ✅ **Intelligent Topic Rotation**: Performance-based topic selection
- ✅ **Natural Posting Behavior**: Anti-bot detection এর জন্য randomized timing
- ✅ **Engagement Learning**: Post performance track করে future optimization
- ✅ **Comprehensive Analytics**: Detailed engagement tracking এবং insights

### Advanced Features  
- 🤖 **Human-like Behavior**: Random delays, varied content, mixed formats
- 📊 **Performance Analytics**: Topic-wise engagement analysis
- 🎯 **Smart Scheduling**: Multiple posting windows with randomization
-  **Engagement Prediction**: AI-powered performance forecasting
- 🔄 **Topic Learning**: Adaptive topic selection based on engagement

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- LinkedIn API Access


### Installation

1. **Clone and Setup**
```bash
git clone <your-repo>
cd linkedin_ai_poster

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
# Copy template and configure
cp .env.template .env

# Edit .env file with your API keys
nano .env
```

3. **Required API Keys**

**OpenAI API Key** (Required)
- Visit: https://platform.openai.com/api-keys
- Create new API key
- Add to `.env`: `OPENAI_API_KEY=your_key_here`

**LinkedIn API** (Required)
- Follow LinkedIn API documentation for OAuth setup
- Get access token and person ID
- Add to `.env`: 
  ```
  LINKEDIN_ACCESS_TOKEN=your_token
  LINKEDIN_PERSON_ID=your_person_id
  ```

4. **Run the Application**
```bash
# Development mode
python app/main.py

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Quick Test
```bash
# Test post generation (without publishing)
curl http://localhost:8000/debug/generate-post?topic=Programming

# Manual generate (creates pending post + sends approval email)
curl -X POST http://localhost:8000/generate-post \
  -H "Content-Type: application/json" \
  -d '{"topic": "Programming"}'

# Check system status  
curl http://localhost:8000/health
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed system status
- `GET /dashboard` - Complete analytics dashboard

### Posting
- `POST /generate-post` - Generate a post and send approval email
- `POST /post/manual` - Backward-compatible alias for approval flow
- `GET /approve-post/{post_id}?token=...` - Approve and publish
- `GET /reject-post/{post_id}?token=...` - Reject and stop publishing
- `GET /approval-form/{post_id}?token=...` - Optional image URL/upload before approve
- `POST /analytics/update` - Update engagement data

### Topic Management
- `GET /topics/insights` - Topic performance analysis
- `GET /topics/recommended` - Get recommended topics

### Scheduler Control
- `GET /scheduler/status` - Scheduler status
- `POST /scheduler/start` - Start automatic posting
- `POST /scheduler/stop` - Stop automatic posting

### Analytics
- `GET /analytics/engagement` - Engagement dashboard
- `GET /analytics/predict?topic=X` - Performance prediction

## ⚙️ Configuration

### Approval and Email
```env
BASE_URL=http://localhost:8000
APPROVAL_SECRET=change-this-to-a-long-random-secret
APPROVAL_TOKEN_EXPIRES_HOURS=24

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_smtp_username
SMTP_PASSWORD=your_smtp_password
SMTP_USE_TLS=true

EMAIL_FROM=noreply@example.com
APPROVAL_EMAIL_TO=approver@example.com
```

### Posting Schedule
```env
MAX_POSTS_PER_DAY=2
MIN_HOURS_BETWEEN_POSTS=4
TIMEZONE=Asia/Dhaka
```

### Natural Behavior
```env
SKIP_POST_PROBABILITY=0.15    # 15% chance to skip scheduled posts
DOUBLE_POST_PROBABILITY=0.05  # 5% chance for extra posts
```

### Posting Windows
Default windows (randomized):
- **Morning**: 9:30-10:30 AM
- **Evening**: 7:30-8:30 PM  
- **Afternoon**: 2:00-3:00 PM (lower priority)

## 🧠 How It Works

### 1. Topic Selection Engine
**Topics Covered:**
- Artificial Intelligence
- Software Engineering  
- Web Development
- Programming
- Backend Development
- DevOps
- Debugging Lessons
- Developer Productivity
- And more...

### 2. Content Generation
**Content Features:**
- 50-120 words in Bengali
- Conversational এবং authentic tone
- Professional যেতে friendly
- Real developer experience এর মতো
- Natural questions এবং engagement hooks

### 3. Publishing & Analytics
Publishes to LinkedIn and tracks performance for future optimization.

## 📈 Analytics & Learning

### Engagement Scoring
```
Score = likes + (comments × 3) + impression_efficiency
```

### Topic Learning
- High-performing topics get higher selection probability
- Declining topics get reduced frequency  
- Unused topics get periodic chances
- Balanced approach prevents over-reliance

## 🔒 Anti-Bot Features

### Natural Behavior Simulation
- **Random Timing**: ±5-10 minute jitter on scheduled posts
- **Content Variation**: Multiple post templates and formats
- **Skip Probability**: Occasionally skip scheduled posts
- **Variable Frequency**: Not every single day posting
- **Natural Delays**: Realistic delays between actions

### Detection Avoidance
- **Humanized Content**: Varied writing styles and formats
- **Mixed Posting Patterns**: Different times and frequencies
- **Authentic Engagement**: Real developer voice এবং experiences

## 🛠️ Development

### Project Structure
```
linkedin_ai_poster/
├── ai/                     # AI providers
├── app/                    # FastAPI application  
├── database/              # Database models
├── services/             # Core services
│   ├── topic_engine.py      # Topic selection logic
│   ├── post_generator.py    # Content generation
│   ├── linkedin_publisher.py # LinkedIn API
│   └── engagement_engine.py  # Analytics & learning
├── scheduler/            # Automated scheduling
├── utils/               # Utilities
└── requirements.txt
```

## 🚀 Deployment

### Local Development
```bash
python app/main.py
```

### Production (Gunicorn)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Heroku Deployment
```bash
# Already configured with Procfile
git push heroku main
```

## 🔧 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY
```

**"LinkedIn API authentication failed"**  
- Verify access token is not expired
- Check LinkedIn API permissions
- Ensure person ID is correct

### Debug Mode
```env  
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

---

**Made with ❤️ for Bengali developers**
   ```
   pip install -r requirements.txt
   ```
3. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

## Workflow
- Scheduler triggers post generation
- AI generates post
- Email sent for approval
- Approve via link
- Post published to LinkedIn

## Environment Variables
- GEMINI_KEYS
- OPENROUTER_KEYS
- OPENAI_KEYS
- LINKEDIN_ACCESS_TOKEN
- LINKEDIN_PERSON_URN
- EMAIL_ADDRESS
- EMAIL_PASSWORD

## Logging & Error Handling
- Logs all actions
- Retries and fallback for AI providers
- Robust error handling

---

**Replace placeholder values in `.env` with your actual credentials.**
