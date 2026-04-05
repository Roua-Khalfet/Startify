"""
Serializers for ComplianceGuard API.
"""

from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField()
    project_context = serializers.CharField(required=False, allow_blank=True)
    sector = serializers.CharField(required=False, allow_blank=True)
    mode = serializers.CharField(required=False, default="kb", allow_blank=True)


class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()
    sources = serializers.ListField(child=serializers.CharField())
    source_type = serializers.CharField()


class ConformiteRequestSerializer(serializers.Serializer):
    project_description = serializers.CharField(required=False, default="", allow_blank=True)
    sector = serializers.CharField(required=False, default="SaaS", allow_blank=True)
    capital = serializers.IntegerField(required=False, default=None, allow_null=True)
    type_societe = serializers.CharField(required=False, default="SUARL", allow_blank=True)


class ConformiteCritereSerializer(serializers.Serializer):
    label = serializers.CharField()
    score = serializers.IntegerField()
    status = serializers.CharField()
    article = serializers.CharField()
    article_source = serializers.CharField()
    details = serializers.CharField()
    category = serializers.CharField(required=False, default="")
    recommendation = serializers.CharField(required=False, allow_null=True, default=None)


class RiskProfileSerializer(serializers.Serializer):
    niveau = serializers.CharField()
    autorisations_requises = serializers.ListField(child=serializers.CharField())
    capital_recommande = serializers.IntegerField()
    delai_conformite = serializers.CharField()


class ConformiteResponseSerializer(serializers.Serializer):
    score_global = serializers.IntegerField()
    status = serializers.CharField()
    criteres = ConformiteCritereSerializer(many=True)
    risk_profile = RiskProfileSerializer(required=False)
    recommendations = serializers.ListField(child=serializers.CharField(), required=False)
    lois_applicables = serializers.ListField(child=serializers.CharField(), required=False)


class DocumentRequestSerializer(serializers.Serializer):
    doc_type = serializers.ChoiceField(
        choices=["statuts", "cgu", "contrat_investissement", "demande_label", "all"]
    )
    nom_startup = serializers.CharField()
    activite = serializers.CharField()
    fondateurs = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )
    capital_social = serializers.IntegerField(required=False, default=1000)
    siege_social = serializers.CharField(required=False, default="Tunis")
    type_societe = serializers.CharField(required=False, default="SUARL")


class DocumentResponseSerializer(serializers.Serializer):
    doc_type = serializers.CharField()
    content = serializers.CharField()
    filename = serializers.CharField()


class GraphNodeSerializer(serializers.Serializer):
    id = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    properties = serializers.DictField(required=False, default=dict)


class GraphEdgeSerializer(serializers.Serializer):
    source = serializers.CharField()
    target = serializers.CharField()
    relation = serializers.CharField()


class GraphResponseSerializer(serializers.Serializer):
    nodes = GraphNodeSerializer(many=True)
    edges = GraphEdgeSerializer(many=True)


class VeilleItemSerializer(serializers.Serializer):
    url = serializers.CharField()
    nom = serializers.CharField()
    last_check = serializers.CharField()
    has_changed = serializers.BooleanField()
    status = serializers.CharField()


class VeilleResponseSerializer(serializers.Serializer):
    items = VeilleItemSerializer(many=True)
    last_update = serializers.CharField()


class SuggestionsRequestSerializer(serializers.Serializer):
    project_description = serializers.CharField()
    sector = serializers.CharField()


class SuggestionsResponseSerializer(serializers.Serializer):
    questions = serializers.ListField(child=serializers.CharField())
