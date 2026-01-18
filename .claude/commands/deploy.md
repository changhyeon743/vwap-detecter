# Deploy to Server

Deploy changes to the crypto server.

## Instructions

1. If there are uncommitted changes, ask for commit message then `git add -A && git commit -m "message"`
2. `git push`
3. `ssh crypto "cd /root/vwap-detector && git pull"`
4. If `main.py` was changed: `ssh crypto "tmux send-keys -t vwap C-c"` then `ssh crypto "tmux send-keys -t vwap 'python3 main.py' Enter"`
