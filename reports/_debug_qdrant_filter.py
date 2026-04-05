from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

client = QdrantClient(url='http://localhost:6333')
flt = qmodels.Filter(must=[
    qmodels.FieldCondition(key='source_file', match=qmodels.MatchValue(value='test.pdf')),
    qmodels.FieldCondition(key='chunk_index', match=qmodels.MatchValue(value=2)),
])
res = client.scroll(collection_name='user_uploads', scroll_filter=flt, limit=2, with_payload=True, with_vectors=False)
print(type(res))
print(res)
