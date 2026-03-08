from app.routes.health import router as health_router
from app.routes.cluster import router as cluster_router
from app.routes.chat import router as chat_router
from app.routes.monitoring import router as monitoring_router

__all__ = [
    'health_router',
    'cluster_router', 
    'chat_router',
    'monitoring_router'
]