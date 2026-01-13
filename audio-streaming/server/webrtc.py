# server/webrtc.py
from aiortc import RTCPeerConnection

sessions: dict[str, RTCPeerConnection] = {}

async def create_session():
    session_id = __import__("uuid").uuid4().hex
    pc = RTCPeerConnection()
    sessions[session_id] = pc
    return session_id, pc

def get_pc(session_id: str):
    return sessions.get(session_id)

def close_session(session_id: str):
    pc = sessions.pop(session_id, None)
    if pc:
        async def _close():
            await pc.close()
        __import__("asyncio").create_task(_close())