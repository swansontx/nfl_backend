#!/usr/bin/env python3
"""
Expose the API publicly using ngrok tunnel

This creates a public URL that can be shared with other agents/services.
"""

from pyngrok import ngrok
import time

# Kill any existing tunnels
ngrok.kill()

# Start tunnel to port 8000
print("=" * 80)
print("STARTING NGROK TUNNEL")
print("=" * 80)
print()

try:
    # Create tunnel
    public_url = ngrok.connect(8000, bind_tls=True)

    print(f"✅ API is now publicly accessible!")
    print()
    print(f"Public URL: {public_url}")
    print()
    print("Share this URL with your frontend agent:")
    print("-" * 80)
    print(f"Base URL: {public_url}")
    print(f"Swagger UI: {public_url}/docs")
    print(f"Health check: {public_url}/health")
    print()
    print(f"Example endpoints:")
    print(f"  {public_url}/api/v1/games/?season=2024")
    print(f"  {public_url}/api/v1/recommendations/{{game_id}}")
    print()
    print("-" * 80)
    print()
    print("Press Ctrl+C to stop the tunnel")
    print()

    # Keep tunnel alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping tunnel...")
        ngrok.kill()
        print("✅ Tunnel stopped")

except Exception as e:
    print(f"❌ Error creating tunnel: {e}")
    print()
    print("You may need to sign up for a free ngrok account:")
    print("1. Go to https://ngrok.com/")
    print("2. Sign up for free")
    print("3. Get your auth token")
    print("4. Run: ngrok authtoken YOUR_TOKEN")
