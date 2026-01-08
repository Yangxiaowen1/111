from django.contrib import admin

from .models import (
    UserProfile,
    UserClickBehavior,
    UserDetailViewBehavior,
    UserFavoriteBehavior,
    UserCommentBehavior,
)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "username", "nickname", "city", "gender", "created_at")
    search_fields = ("username", "nickname", "city")
    list_filter = ("gender", "city")


@admin.register(UserClickBehavior)
class UserClickBehaviorAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "poi_id", "scene", "occurred_at")
    search_fields = ("user__username", "poi_id")
    list_filter = ("scene",)


@admin.register(UserDetailViewBehavior)
class UserDetailViewBehaviorAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "poi_id", "stay_seconds", "occurred_at")
    search_fields = ("user__username", "poi_id")


@admin.register(UserFavoriteBehavior)
class UserFavoriteBehaviorAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "poi_id", "is_favorite", "occurred_at")
    list_filter = ("is_favorite",)


@admin.register(UserCommentBehavior)
class UserCommentBehaviorAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "poi_id", "rating", "occurred_at")
    search_fields = ("user__username", "poi_id", "content")
