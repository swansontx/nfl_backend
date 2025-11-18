from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='NFL Props Backend API')

@app.get('/health')
async def health():
    return {'status': 'ok'}

class RecomputeRequest(BaseModel):
    game_id: str

@app.post('/admin/recompute')
async def recompute(req: RecomputeRequest):
    # placeholder: trigger recompute orchestration for a game
    return {'status': 'started', 'game_id': req.game_id}

@app.get('/game/{game_id}/projections')
async def get_projections(game_id: str):
    # placeholder: load projections from outputs/ and return filtered results
    return {'game_id': game_id, 'projections': []}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
