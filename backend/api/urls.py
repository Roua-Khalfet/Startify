"""
API URLs for ComplianceGuard.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_root, name="api-root"),
    path("chat/", views.chat, name="chat"),
    path("chat/knowledge/", views.chat_knowledge, name="chat-knowledge"),
    path("upload/", views.upload_document, name="upload"),
    path("conformite/", views.conformite, name="conformite"),
    path("documents/", views.generate_documents, name="documents"),
    path("graph/", views.get_graph, name="graph"),
    path("veille/", views.get_veille, name="veille"),
    path("suggestions/", views.get_suggestions, name="suggestions"),
]
