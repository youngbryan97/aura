import unittest
from unittest.mock import AsyncMock

from core.security.permission_guard import PermissionGuard, PermissionType


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
