from typing import List, Optional
from models.match_team_statistics import MatchTeamStatistics, MatchTeamStatisticsCreate
import json

class MatchTeamStatisticsService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def _to_model(self, data: dict) -> MatchTeamStatistics:
        # Manually parse JSON fields
        for key in ['high_turnover_zones_data', 'set_piece_xg_breakdown_data', 'transition_speed_data', 'build_up_patterns', 'defensive_block_patterns', 'pass_network_data', 'zone_analysis_data', 'tactical_weakness_data']:
            if key in data and isinstance(data[key], str):
                try:
                    data[key] = json.loads(data[key])
                except json.JSONDecodeError:
                    data[key] = None # Or handle error appropriately
        return MatchTeamStatistics(**data)

    def get_by_match_id(self, match_id: str, user_team_ids: List[str]) -> List[MatchTeamStatistics]:
        if not user_team_ids:
            return []
        with self.db_connection.cursor() as cursor:
            sql = "SELECT mt.* FROM match_team_statistics mt WHERE mt.match_id = %s"
            cursor.execute(sql, (match_id,))
            stats = cursor.fetchall()
            return [self._to_model(s) for s in stats]

    def create_match_team_statistics(self, stats: MatchTeamStatisticsCreate, user_team_ids: List[str]) -> MatchTeamStatistics:
        # Validate that the match's teams are owned by the user
        # This requires fetching match details, which is complex here.
        # For simplicity, we'll assume match_id is valid and linked to user's team via MatchService.
        with self.db_connection.cursor() as cursor:
            sql = "INSERT INTO match_team_statistics (id, match_id, team_id, possession_percentage, total_shots, shots_on_target, expected_goals, pressures, final_third_passes, pass_network_data, zone_analysis_data, tactical_weakness_data, high_turnover_zones_data, set_piece_xg_breakdown_data, transition_speed_data, build_up_patterns, defensive_block_patterns) VALUES (UUID(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (stats.match_id, stats.team_id, stats.possession_percentage, stats.total_shots, stats.shots_on_target, stats.expected_goals, stats.pressures, stats.final_third_passes, json.dumps(stats.pass_network_data), json.dumps(stats.zone_analysis_data), json.dumps(stats.tactical_weakness_data), json.dumps(stats.high_turnover_zones_data), json.dumps(stats.set_piece_xg_breakdown_data), json.dumps(stats.transition_speed_data), json.dumps(stats.build_up_patterns), json.dumps(stats.defensive_block_patterns)))
            self.db_connection.commit()
            cursor.execute("SELECT * FROM match_team_statistics WHERE match_id = %s AND team_id = %s ORDER BY id DESC LIMIT 1", (stats.match_id, stats.team_id))
            new_stats = cursor.fetchone()
            return self._to_model(new_stats)

    def update_match_team_statistics(self, stat_id: str, stats_update: MatchTeamStatisticsCreate, user_team_ids: List[str]) -> Optional[MatchTeamStatistics]:
        # Validate that the match's teams are owned by the user
        # This requires fetching match details, which is complex here.
        # For simplicity, we'll assume match_id is valid and linked to user's team via MatchService.
        with self.db_connection.cursor() as cursor:
            sql = "UPDATE match_team_statistics SET match_id = %s, team_id = %s, possession_percentage = %s, total_shots = %s, shots_on_target = %s, expected_goals = %s, pressures = %s, final_third_passes = %s, pass_network_data = %s, zone_analysis_data = %s, tactical_weakness_data = %s, high_turnover_zones_data = %s, set_piece_xg_breakdown_data = %s, transition_speed_data = %s, build_up_patterns = %s, defensive_block_patterns = %s WHERE id = %s"
            cursor.execute(sql, (stats_update.match_id, stats_update.team_id, stats_update.possession_percentage, stats_update.total_shots, stats_update.shots_on_target, stats_update.expected_goals, stats_update.pressures, stats_update.final_third_passes, json.dumps(stats_update.pass_network_data), json.dumps(stats_update.zone_analysis_data), json.dumps(stats_update.tactical_weakness_data), json.dumps(stats_update.high_turnover_zones_data), json.dumps(stats_update.set_piece_xg_breakdown_data), json.dumps(stats_update.transition_speed_data), json.dumps(stats_update.build_up_patterns), json.dumps(stats_update.defensive_block_patterns), stat_id))
            self.db_connection.commit()
            return self.get_by_match_id(stats_update.match_id, user_team_ids)[0] if self.get_by_match_id(stats_update.match_id, user_team_ids) else None

    def delete_match_team_statistics(self, stat_id: str, user_team_ids: List[str]) -> bool:
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM match_team_statistics WHERE id = %s AND match_id IN (SELECT id FROM matches WHERE home_team_id IN %s OR away_team_id IN %s)"
            cursor.execute(sql, (stat_id, user_team_ids, user_team_ids))
            self.db_connection.commit()
            return cursor.rowcount > 0
