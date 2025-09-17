# tracking/roster.py
"""
Team-local jersey numbering for soccer tracking.

This module assigns and maintains team-local numbers (1..11, then 12, 13, ...)
for players in TEAM1 and TEAM2 based on currently visible track IDs.

Key rules:
- Only **visible** track IDs of the current frame are considered.
- Numbers are **immediately released** when a track is not visible anymore
  or no longer belongs to TEAM1/TEAM2 (e.g., UNK/REF).
- When a player reappears in a team, the **lowest free number** is assigned.
- If a team has more than 11 visible players, numbering continues with 12, 13, ...

Data structures (per team):
- tid_to_num: maps track id (tid) -> assigned jersey number (int)
- num_to_tid: maps jersey number (int) -> current track id (tid)
"""

from typing import Dict, Tuple, Iterable, Optional


class TeamRoster:
    """
    Manage team-local jersey numbers for TEAM1 and TEAM2.

    Attributes:
        max_players (int): Preferred jersey range per team (1..max_players).
        teams (dict): Per-team state:
            {
              "TEAM1": {"tid_to_num": {tid:int}, "num_to_tid": {num:int}},
              "TEAM2": {"tid_to_num": {tid:int}, "num_to_tid": {num:int}}
            }
    """

    def __init__(self, max_players: int = 11):
        """
        Initialize the roster.

        Args:
            max_players (int): Preferred max number per team (1..max_players).
        """
        self.max_players: int = int(max_players)
        self.teams: Dict[str, Dict[str, Dict[int, int]]] = {
            "TEAM1": {"tid_to_num": {}, "num_to_tid": {}},
            "TEAM2": {"tid_to_num": {}, "num_to_tid": {}},
        }

    def _release_tid(self, team: str, tid: int) -> None:
        """
        Release any jersey number currently assigned to a given track id in a team.

        Args:
            team (str): Team name ("TEAM1" or "TEAM2").
            tid (int): Track id to release.

        Returns:
            None

        Behavior:
            - If the tid has a number in this team, remove both mappings.
        """
        data = self.teams[team]
        if tid in data["tid_to_num"]:
            number = data["tid_to_num"].pop(tid)
            # Remove the reverse mapping if present
            if number in data["num_to_tid"]:
                data["num_to_tid"].pop(number, None)

    def _assign_lowest_free(self, team: str, tid: int) -> int:
        """
        Assign the lowest currently free jersey number to a given track id in a team.

        Args:
            team (str): Team name ("TEAM1" or "TEAM2").
            tid (int): Track id to assign a number to.

        Returns:
            int: The jersey number assigned to the track id.

        Behavior:
            - Prefer a free number in range [1..max_players].
            - If all are taken, assign the smallest free integer > max_players (e.g., 12, 13, ...).
        """
        data = self.teams[team]
        used_numbers = set(data["num_to_tid"].keys())

        # Try to assign from the preferred range first (1..max_players)
        for number in range(1, self.max_players + 1):
            if number not in used_numbers:
                data["num_to_tid"][number] = tid
                data["tid_to_num"][tid] = number
                return number

        # Otherwise, continue above max_players until a free number is found
        number = self.max_players + 1
        while number in used_numbers:
            number += 1

        data["num_to_tid"][number] = tid
        data["tid_to_num"][tid] = number
        return number

    def update(self, frame_idx: int, roles: Dict[int, Dict[str, str]], active_tids: Optional[Iterable[int]]) -> Dict[int, Tuple[str, int]]:
        """
        Update jersey numbers for the current frame and return a tid -> (team, number) map.

        Args:
            frame_idx (int): Current frame index (kept for interface consistency; not used).
            roles (dict): Cluster roles per tid, e.g., { tid: {"role": "TEAM1"|"TEAM2"|"REF"|"UNK"|"BALL"} }.
            active_tids (Iterable[int] | None): Track ids visible in the current frame.

        Returns:
            dict[int, (str, int)]: Mapping { tid: (team, number) } for TEAM1 and TEAM2 only.

        Behavior:
            - Consider only **visible** track ids (active_tids).
            - For each team, compute the set of visible tids with role TEAM1/TEAM2.
            - Release numbers of tids that are no longer visible in their team.
            - Prevent cross-team duplicates by releasing the same tid in the other team.
            - Assign the lowest free number to each visible tid without a number.
            - Return the current (team, number) for all visible tids in TEAM1/TEAM2.
        """
        # Normalize active set
        if active_tids is None:
            active_set = set()
        else:
            active_set = set(int(t) for t in active_tids)

        # Build active tids per team by intersecting visibility and roles
        active_per_team = {"TEAM1": set(), "TEAM2": set()}
        for tid in active_set:
            info = roles.get(tid)
            if info is None:
                continue
            role_value = info.get("role")
            if role_value == "TEAM1":
                active_per_team["TEAM1"].add(tid)
            elif role_value == "TEAM2":
                active_per_team["TEAM2"].add(tid)

        # Release numbers for tids that are not visible in their assigned team anymore
        for team in ("TEAM1", "TEAM2"):
            current_assigned_tids = list(self.teams[team]["tid_to_num"].keys())
            for tid in current_assigned_tids:
                if tid not in active_per_team[team]:
                    self._release_tid(team, tid)

        # Make sure a tid cannot be assigned in both teams simultaneously
        for team in ("TEAM1", "TEAM2"):
            other_team = "TEAM2" if team == "TEAM1" else "TEAM1"
            for tid in active_per_team[team]:
                # If the same tid was previously assigned in the other team, release it there
                self._release_tid(other_team, tid)

        # Assign numbers to visible tids without one and build the result map
        result: Dict[int, Tuple[str, int]] = {}
        for team in ("TEAM1", "TEAM2"):
            for tid in active_per_team[team]:
                if tid not in self.teams[team]["tid_to_num"]:
                    self._assign_lowest_free(team, tid)
                number = self.teams[team]["tid_to_num"][tid]
                result[tid] = (team, number)

        return result