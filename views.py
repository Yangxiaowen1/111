import json
from typing import Optional

from django.contrib.auth import authenticate, get_user_model
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    UserClickBehavior,
    UserCommentBehavior,
    UserDetailViewBehavior,
    UserFavoriteBehavior,
    UserProfile,
)
from .serializers import RegisterSerializer, UserProfileSerializer, UserSerializer

User = get_user_model()


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response(
            {
                "token": token.key,
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        user = authenticate(username=username, password=password)
        if not user:
            return Response(
                {"detail": "用户名或密码错误"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        token, _ = Token.objects.get_or_create(user=user)
        return Response(
            {
                "token": token.key,
                "user": UserSerializer(user).data,
            }
        )


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        """获取当前用户信息"""
        profile, _ = UserProfile.objects.get_or_create(
            username=request.user.username,
            defaults={"email": request.user.email or ""},
        )
        return Response(UserProfileSerializer(profile).data)

    def put(self, request):
        """更新当前用户信息（完整更新）"""
        profile, _ = UserProfile.objects.get_or_create(
            username=request.user.username,
            defaults={"email": request.user.email or ""},
        )
        serializer = UserProfileSerializer(profile, data=request.data, partial=False)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


def resolve_profile(user_id: Optional[int], username: Optional[str] = None) -> UserProfile:
    if user_id:
        return get_object_or_404(UserProfile, pk=int(user_id))
    if username:
        return get_object_or_404(UserProfile, username=username)
    profile, _ = UserProfile.objects.get_or_create(
        username="demo_user",
        defaults={"nickname": "体验用户"},
    )
    return profile


class ClickView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        profile = resolve_profile(request.data.get("user_id"), request.data.get("username"))
        poi_id = request.data.get("poi_id")
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        record = UserClickBehavior.objects.create(
            user=profile,
            poi_id=int(poi_id),
            scene=request.data.get("scene", "attraction_list"),
            device=request.data.get("device", "web"),
            exposure_id=request.data.get("exposure_id", ""),
            occurred_at=timezone.now(),
        )
        return Response({"record_id": record.id}, status=status.HTTP_201_CREATED)


class FavoriteView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        profile = resolve_profile(
            request.query_params.get("user_id"),
            request.query_params.get("username"),
        )
        poi_id = request.query_params.get("poi_id")
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        latest = (
            UserFavoriteBehavior.objects.filter(user=profile, poi_id=int(poi_id))
            .order_by("-occurred_at")
            .first()
        )
        return Response(
            {"is_favorite": bool(latest and latest.is_favorite)}
        )

    def post(self, request):
        profile = resolve_profile(request.data.get("user_id"), request.data.get("username"))
        poi_id = request.data.get("poi_id")
        is_favorite = request.data.get("is_favorite", True)
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        record = UserFavoriteBehavior.objects.create(
            user=profile,
            poi_id=int(poi_id),
            is_favorite=bool(is_favorite),
            scene=request.data.get("scene", "frontend"),
            device=request.data.get("device", "web"),
            occurred_at=timezone.now(),
        )
        return Response(
            {"is_favorite": record.is_favorite, "record_id": record.id},
            status=status.HTTP_201_CREATED,
        )


class DetailViewLogView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        profile = resolve_profile(request.data.get("user_id"), request.data.get("username"))
        poi_id = request.data.get("poi_id")
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        record = UserDetailViewBehavior.objects.create(
            user=profile,
            poi_id=int(poi_id),
            scene=request.data.get("scene", "detail_page"),
            device=request.data.get("device", "web"),
            occurred_at=timezone.now(),
            stay_seconds=int(request.data.get("stay_seconds") or 0),
        )
        return Response({"record_id": record.id}, status=status.HTTP_201_CREATED)


class CommentView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        poi_id = request.query_params.get("poi_id")
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        queryset = (
            UserCommentBehavior.objects.filter(poi_id=int(poi_id))
            .select_related("user")
            .order_by("-occurred_at")
        )
        results = [
            {
                "id": item.id,
                "user": item.user.username,
                "nickname": item.user.nickname,
                "rating": item.rating,
                "content": item.content,
                "images": json.loads(item.image_urls or "[]"),
                "occurred_at": item.occurred_at,
            }
            for item in queryset
        ]
        return Response(results)

    def post(self, request):
        profile = resolve_profile(request.data.get("user_id"), request.data.get("username"))
        poi_id = request.data.get("poi_id")
        if not poi_id:
            return Response({"detail": "缺少 poi_id 参数"}, status=status.HTTP_400_BAD_REQUEST)
        images = request.data.get("images") or []
        record = UserCommentBehavior.objects.create(
            user=profile,
            poi_id=int(poi_id),
            rating=int(request.data.get("rating") or 0),
            content=request.data.get("content", ""),
            image_urls=json.dumps(images, ensure_ascii=False),
            scene=request.data.get("scene", "detail_page"),
            device=request.data.get("device", "web"),
            occurred_at=timezone.now(),
        )
        return Response(
            {
                "id": record.id,
                "user": record.user.username,
                "nickname": record.user.nickname,
                "rating": record.rating,
                "content": record.content,
                "images": images,
                "occurred_at": record.occurred_at,
            },
            status=status.HTTP_201_CREATED,
        )

    def patch(self, request):
        """更新当前用户信息（部分更新）"""
        profile, _ = UserProfile.objects.get_or_create(
            username=request.user.username,
            defaults={"email": request.user.email or ""},
        )
        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)
