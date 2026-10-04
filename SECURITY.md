# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The Turing team takes the security and integrity of causal discovery algorithms and user data seriously.

If you discover a security vulnerability, please report it responsibly:

1. **Do not create a public GitHub issue.**
2. Email security findings to **security@turing-platform.io** (or create a private GitHub Security Advisory).
3. Include detailed steps to reproduce the issue, proof-of-concept payloads (if applicable), and impacted endpoints.

### Expected Response Times
- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 72 hours
- **Patch Release & Advisory**: Within 7–14 business days depending on severity

## Security Best Practices for Deployments

- **API Keys**: Never commit production LLM keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`) to version control. Use `.env.local` or environment secrets.
- **CORS Configuration**: In production, always set `CORS_ORIGINS` to the exact domain(s) hosting your frontend application rather than wildcards.
- **File Upload Limits**: Ensure reverse proxies (e.g. Nginx, Cloudflare) enforce reasonable body size limits (default recommended: ≤ 50MB) to mitigate denial-of-service risks.
- **Database Access**: Ensure PostgreSQL and Redis instances are deployed in private container networks without direct exposure to the public internet.
