from __future__ import annotations

from django.contrib.auth import get_user_model
from rest_framework import serializers

from .models import UserProfile

User = get_user_model()


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, min_length=6)
    email = serializers.EmailField(required=False, allow_blank=True)

    def validate_username(self, value: str) -> str:
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("用户名已存在")
        return value

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email", ""),
            password=validated_data["password"],
        )


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email", "first_name", "last_name")


class UserProfileSerializer(serializers.ModelSerializer):
    """用户信息序列化器"""

    class Meta:
        model = UserProfile
        fields = (
            "id",
            "external_id",
            "username",
            "nickname",
            "email",
            "phone",
            "gender",
            "city",
            "avatar",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "external_id", "username", "created_at", "updated_at")

