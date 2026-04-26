import unittest
import time
from unittest.mock import AsyncMock

import core.security.permission_guard as permission_guard_module
from core.security.permission_guard import PermissionGuard, PermissionType, get_permission_guard


class TestPermissionGuardCache(unittest.IsolatedAsyncioTestCase):
    async def test_force_refresh_reuses_fresh_cache(self):
        guard = PermissionGuard()
        guard._force_refresh_floor_s = 60.0
        guard._check_screen_permission = AsyncMock(
            return_value={"granted": True, "status": "active", "guidance": ""}
        )

        first = await guard.check_permission(PermissionType.SCREEN, force=True)
        second = await guard.check_permission(PermissionType.SCREEN, force=True)

        self.assertEqual(first, second)
        guard._check_screen_permission.assert_awaited_once()

    async def test_non_force_refreshes_stale_cache_after_ttl(self):
        guard = PermissionGuard()
        guard._cache_ttl_s = 5.0
        guard._cache[PermissionType.SCREEN] = {
            "granted": False,
            "status": "denied",
            "guidance": "stale",
        }
        guard._cache_ts[PermissionType.SCREEN] = time.monotonic() - 10.0
        guard._check_screen_permission = AsyncMock(
            return_value={"granted": True, "status": "active", "guidance": ""}
        )

        refreshed = await guard.check_permission(PermissionType.SCREEN, force=False)

        self.assertTrue(refreshed["granted"])
        guard._check_screen_permission.assert_awaited_once()

    def test_shared_permission_guard_accessor_reuses_singleton(self):
        original = permission_guard_module._SHARED_PERMISSION_GUARD
        permission_guard_module._SHARED_PERMISSION_GUARD = None
        try:
            first = get_permission_guard()
            second = get_permission_guard()
        finally:
            permission_guard_module._SHARED_PERMISSION_GUARD = original

        self.assertIs(first, second)
