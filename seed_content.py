#!/usr/bin/env python3
"""
Seed the content database with sample tech news items for testing.
এই script দিয়ে database এ sample content add করুন testing এর জন্য।
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

# Sample content items
SAMPLE_CONTENT = [
    {
        "title": "Python 3.14 Released with Major Performance Improvements",
        "url": "https://example.com/python-3-14-release",
        "source": "hackernews",
        "author": "Guido van Rossum",
        "summary": "The latest version of Python brings significant performance enhancements and new features. The interpreter is now 40% faster than Python 3.10.",
        "full_text": "Python 3.14 has been officially released with groundbreaking performance improvements. The development team has optimized the bytecode interpreter, resulting in a 40% speed boost compared to Python 3.10. New features include pattern matching enhancements, improved async/await, and better type annotations. The release also introduces several breaking changes that developers need to be aware of when upgrading their codebases.",
        "key_points": ["40% faster interpreter", "Enhanced pattern matching", "Better async/await", "Improved type hints"],
        "tags": ["python", "performance", "programming", "release"],
        "score": 95.0,
        "external_score": 2500,
    },
    {
        "title": "Kubernetes 1.30: What's New in the Latest Release",
        "url": "https://example.com/kubernetes-1-30",
        "source": "devto",
        "author": "Sarah Chen",
        "summary": "Kubernetes 1.30 brings several improvements including better resource management and security enhancements.",
        "full_text": "The Kubernetes community has released version 1.30 with several exciting new features. The latest release focuses on improving resource management, security, and developer experience. Container orchestration is now more efficient, with better scheduling algorithms and improved load balancing.",
        "key_points": ["New scheduling algorithms", "Enhanced security", "Better load balancing", "Improved UI"],
        "tags": ["kubernetes", "devops", "cloud", "containers"],
        "score": 88.0,
        "external_score": 1800,
    },
    {
        "title": "React 19: React Compiler Goes Production",
        "url": "https://example.com/react-19-compiler",
        "source": "techcrunch",
        "author": "Dan Abramov",
        "summary": "React's new compiler is now ready for production use, offering automatic optimization without developer intervention.",
        "full_text": "The React team has announced that the React Compiler is now production-ready. This automatic optimization tool analyzes your code and applies optimizations without requiring manual intervention. It works with existing React patterns and is fully backward compatible.",
        "key_points": ["Production-ready", "Automatic optimization", "Backward compatible", "No breaking changes"],
        "tags": ["react", "javascript", "frontend", "compiler"],
        "score": 92.0,
        "external_score": 3200,
    },
    {
        "title": "Database Trends 2026: SQL vs NoSQL Evolution",
        "url": "https://example.com/database-trends-2026",
        "source": "reddit",
        "author": "Database Expert",
        "summary": "Exploring the latest trends in database technology and how SQL and NoSQL are converging.",
        "full_text": "As we move through 2026, database technology continues to evolve. The traditional SQL vs NoSQL debate is becoming less relevant as databases adopt features from both paradigms. NewSQL databases are gaining popularity for their ability to combine ACID guarantees with horizontal scalability.",
        "key_points": ["SQL vs NoSQL convergence", "NewSQL popularity", "ACID guarantees", "Horizontal scalability"],
        "tags": ["database", "sql", "nosql", "architecture"],
        "score": 82.0,
        "external_score": 1500,
    },
    {
        "title": "Machine Learning Ops: Best Practices for 2026",
        "url": "https://example.com/mlops-best-practices",
        "source": "hackernews",
        "author": "ML Engineer",
        "summary": "Modern best practices for managing machine learning models in production environments.",
        "full_text": "MLOps has matured significantly in 2026. Organizations are now focusing on reproducibility, monitoring, and efficient model deployment. Key practices include using containerization for consistency, implementing comprehensive monitoring for model performance, and establishing robust version control systems for datasets and models.",
        "key_points": ["Containerization", "Model monitoring", "Version control", "Data management"],
        "tags": ["ai", "machine-learning", "mlops", "devops"],
        "score": 85.0,
        "external_score": 1200,
    },
    {
        "title": "Web Assembly and the Future of Frontend Development",
        "url": "https://example.com/webassembly-future",
        "source": "devto",
        "author": "Frontend Developer",
        "summary": "How WebAssembly is reshaping frontend development and what it means for JavaScript.",
        "full_text": "WebAssembly continues to gain traction as more frameworks adopt it. Languages like Rust and C++ can now be compiled to WebAssembly, enabling high-performance web applications. This doesn't mean JavaScript is going away, but rather that developers have more tools in their toolkit for building performant applications.",
        "key_points": ["WebAssembly adoption", "Rust in web", "Performance gains", "Multi-language support"],
        "tags": ["webassembly", "frontend", "javascript", "performance"],
        "score": 88.0,
        "external_score": 1600,
    },
    {
        "title": "Cloud Security: Essential Practices for 2026",
        "url": "https://example.com/cloud-security-2026",
        "source": "techcrunch",
        "author": "Security Expert",
        "summary": "Essential security practices for cloud-native applications and infrastructure.",
        "full_text": "Cloud security remains a critical concern for organizations. Best practices include implementing Zero Trust architecture, using managed secrets services, and conducting regular security audits. The principle of least privilege should be applied across all cloud resources.",
        "key_points": ["Zero Trust architecture", "Managed secrets", "Regular audits", "Least privilege"],
        "tags": ["security", "cloud", "devops", "architecture"],
        "score": 90.0,
        "external_score": 2100,
    },
    {
        "title": "Building Scalable APIs with FastAPI",
        "url": "https://example.com/fastapi-scalable-apis",
        "source": "reddit",
        "author": "Python Developer",
        "summary": "Learn how to build high-performance, scalable APIs using FastAPI.",
        "full_text": "FastAPI has become a go-to framework for building modern Python APIs. With built-in support for async programming, automatic API documentation, and type validation, FastAPI allows developers to build scalable applications quickly. This guide covers best practices for using FastAPI in production.",
        "key_points": ["Async support", "Auto documentation", "Type validation", "Production ready"],
        "tags": ["fastapi", "python", "api", "backend"],
        "score": 86.0,
        "external_score": 1400,
    },
]

def generate_url_hash(url: str) -> str:
    """Generate a hash for the URL."""
    return hashlib.md5(url.encode()).hexdigest()

def seed_database(db_path: str = "linkedin_ai_poster.db"):
    """Seed the content_items table with sample data."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                url_hash TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                author TEXT,
                full_text TEXT,
                summary TEXT,
                key_points TEXT,
                image_url TEXT,
                tags TEXT,
                score REAL DEFAULT 0,
                external_score INTEGER DEFAULT 0,
                published_at TIMESTAMP,
                fetched_at TIMESTAMP,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_for_post BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        # Insert sample data
        now = datetime.utcnow()
        inserted = 0
        
        for i, item in enumerate(SAMPLE_CONTENT):
            url_hash = generate_url_hash(item["url"])
            published_at = (now - timedelta(days=7-i)).isoformat()
            processed_at = (now - timedelta(days=6-i)).isoformat()
            
            try:
                cursor.execute("""
                    INSERT INTO content_items (
                        title, url, url_hash, source, author,
                        full_text, summary, key_points, tags,
                        score, external_score, published_at, fetched_at,
                        processed_at, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["title"],
                    item["url"],
                    url_hash,
                    item["source"],
                    item["author"],
                    item["full_text"],
                    item["summary"],
                    json.dumps(item["key_points"]),
                    json.dumps(item["tags"]),
                    item["score"],
                    item["external_score"],
                    published_at,
                    (now - timedelta(days=5-i)).isoformat(),
                    processed_at,
                    "processed",
                    (now - timedelta(days=4-i)).isoformat(),
                ))
                inserted += 1
                print(f"✅ Added: {item['title'][:60]}...")
                
            except sqlite3.IntegrityError:
                print(f"⚠️  Duplicate: {item['title'][:60]}...")
        
        conn.commit()
        conn.close()
        
        print(f"\n✨ Successfully seeded {inserted} content items!")
        return inserted
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        raise

if __name__ == "__main__":
    seed_database()
