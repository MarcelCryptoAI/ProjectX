#!/bin/bash

# DEPLOYMENT SCRIPT FOR BYBIT-AI-BOT-EU ONLY!
# This prevents accidentally deploying to wrong app

echo "🚀 DEPLOYING TO BYBIT-AI-BOT-EU (team noml)"
echo "🎯 App: bybit-ai-bot-eu-d0c891a4972a.herokuapp.com"
echo ""

# Verify we're pushing to the correct remote
REMOTE_URL=$(git remote get-url heroku)
if [[ "$REMOTE_URL" != *"bybit-ai-bot-eu"* ]]; then
    echo "❌ ERROR: Wrong Heroku remote!"
    echo "Current remote: $REMOTE_URL"
    echo "Expected: git.heroku.com/bybit-ai-bot-eu.git"
    exit 1
fi

echo "✅ Remote verified: $REMOTE_URL"
echo ""

# Stage all changes
git add -A

# Commit with timestamp
COMMIT_MSG="Deploy $(date '+%Y-%m-%d %H:%M:%S') - bybit-ai-bot-eu

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git commit -m "$COMMIT_MSG"

# Deploy to correct app
echo "🚀 Pushing to bybit-ai-bot-eu..."
git push heroku master

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "🌐 App URL: https://bybit-ai-bot-eu-d0c891a4972a.herokuapp.com/"