# TOTP Setup Guide

## Generate a TOTP Secret

```bash
python3 -c "
import secrets, base64
secret = base64.b32encode(secrets.token_bytes(20)).decode('ascii').rstrip('=')
print(f'TOTP_SECRET={secret}')
"
```

## Generate a Password

```bash
python3 -c "
import secrets
print(f'PASSWORD={secrets.token_urlsafe(16)}')
"
```

## Configure in .env

Generate fresh credentials locally — never commit them.

```bash
# Generate a 16-character password
python3 -c "import secrets; print(secrets.token_urlsafe(16))"

# Generate a base32 TOTP secret
python3 -c "import secrets, base64; print(base64.b32encode(secrets.token_bytes(20)).decode('ascii').rstrip('='))"
```

Add to `.env`:

```bash
MEMORY_OS_PASSWORD=<the password you just generated>
MEMORY_OS_TOTP_SECRET=<the TOTP secret you just generated>
```

> **Security note:** Anyone with the password + secret can log in. Add the
> `.env` file to `.gitignore` (already the default here) and rotate if it
> ever leaks into a public repository.

## Scan with Authenticator App

1. Open your TOTP app (Google Authenticator, Authy, Microsoft Authenticator, etc.)
2. Add account → Enter setup key
3. Account name: `MemoryOS`
4. Key: paste the `MEMORY_OS_TOTP_SECRET` value
5. Type: TOTP (time-based)
6. Algorithm: SHA1 / 30 seconds / 6 digits

Or use this URI (generate a QR code from it):
```
otpauth://totp/MemoryOS:user?secret=REDACTED_TOTP_SECRET&issuer=MemoryOS
```

## Login

1. Go to `https://<your-dashboard-domain>/login` (configure your reverse proxy)
2. Enter password
3. Enter the 6-digit code from your authenticator app
4. Click "登录"

## Token Validity

The TOTP implementation uses a ±1 step window (90 seconds total):
- Current 30s step
- Previous 30s step
- Next 30s step

## Recovery

If you lose access to your authenticator app:

1. SSH into the server
2. Remove `MEMORY_OS_PASSWORD` and `MEMORY_OS_TOTP_SECRET` from `.env`
3. Restart: `systemctl restart openclaw-memory-os`
4. Auth will fall back to the shared `MEMORY_OS_TOKEN` mode
5. You can then regenerate new credentials

## Session Lifetime

Default: 12 hours. To change:
```bash
MEMORY_OS_SESSION_MAX_AGE=259200  # 3 days
```

After the session expires, you will need to re-enter password + TOTP.

## Security Notes

- TOTP implementation uses only Python standard library (HMAC-SHA1)
- No external TOTP packages required
- Secret is stored in `.env` (not committed to git)
- Password hashing: SHA-256 with per-process salt (upgradeable to Argon2id)
