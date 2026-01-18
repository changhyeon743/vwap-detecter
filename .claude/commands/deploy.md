# Deploy to Server

Deploy the current changes to the crypto server.

## Steps

1. Check for uncommitted changes and commit if needed
2. Push to GitHub
3. Pull changes on the server via SSH
4. Analyze git log to recommend which session to restart
5. Restart the specified tmux session (user will enter password)

## Instructions

When running this command:

1. First, check `git status` for any uncommitted changes
2. If there are changes, ask the user for a commit message or suggest one based on the changes
3. Run `git add -A && git commit -m "message"` then `git push`
4. Run `ssh crypto "cd /root/crypto-server && git pull"`
5. **Auto-detect which session to restart** by running `git log -5 --oneline --name-only` and analyzing changed files:
   - Show a table of recent commits with changed files and recommended session
   - Use the file-to-session mapping below to recommend which session(s) to restart
6. Ask user to confirm the recommended session or choose a different one
7. Run `ssh crypto "tmux send-keys -t <session> C-c"` to stop the current process
8. Tell the user to attach to the session and start the process:
   - For vwap: No password needed, just restart

## Session Commands

| Session | Start Command         |
| ------- | --------------------- |
| vwap    | Check the vwap script |

## File-to-Session Mapping

Use this mapping to determine which session needs restart based on changed files:

| File Path Pattern                | Session                  |
| -------------------------------- | ------------------------ |
| `vwap*`                          | vwap                     |
| `.claude/*`, `CLAUDE.md`, `*.md` | None (no restart needed) |

If multiple sessions are affected, list all of them and let the user decide the order.
