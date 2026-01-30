from xfmr_zem.server import ZemServer

app = ZemServer("dummy")

@app.tool()
def echo(message: str) -> dict:
    return {"reply": f"Echo: {message}"}

if __name__ == "__main__":
    app.run()
