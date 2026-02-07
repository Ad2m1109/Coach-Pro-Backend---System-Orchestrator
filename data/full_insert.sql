-- full_insert.sql
-- Insert initial data for soccer_analytics. Run after full_creation.sql

USE soccer_analytics;

-- Define UUIDs for the new user and their team
SET @user_id_new = UUID();
SET @team_id_new = UUID();

-- Insert New User
INSERT INTO users (id, email, password_hash, full_name, is_active) VALUES
(@user_id_new, 'newuser@example.com', '$2b$12$mzKhvh02ijxfifha60gAXexnZzQvhO3.mFC3nT7sE4uiarEf5yj2K', 'New User', TRUE);

-- Insert New Team linked to the New User
INSERT INTO teams (id, name, user_id, primary_color, secondary_color, logo_url) VALUES
(@team_id_new, 'My Awesome Team', @user_id_new, '#0000FF', '#FFFFFF', 'https://cdn.example.com/teams/my_awesome_team.png');

-- Insert Formations
SET @formation_id_433 = UUID();
SET @formation_id_442 = UUID();
SET @formation_id_343 = UUID();
SET @formation_id_352 = UUID();
SET @formation_id_424 = UUID();
SET @formation_id_4231 = UUID();
SET @formation_id_4411 = UUID();
SET @formation_id_4222_narrow = UUID();
SET @formation_id_532 = UUID();
SET @formation_id_541 = UUID();
SET @formation_id_451 = UUID();
SET @formation_id_4141 = UUID();
SET @formation_id_4222_brazil = UUID();
SET @formation_id_3223 = UUID();
SET @formation_id_3313 = UUID();
SET @formation_id_442d = UUID();
INSERT INTO formations (id, name, description, positions, user_id) VALUES
(@formation_id_433, '4-3-3', 'An attacking formation with three forwards.', NULL, NULL),
(@formation_id_442, '4-4-2', 'A classic, balanced formation.', NULL, NULL),
(@formation_id_343, '3-4-3', 'An extremely attacking formation with wing play emphasis.', NULL, NULL),
(@formation_id_352, '3-5-2', 'Attacking formation with strong midfield control.', NULL, NULL),
(@formation_id_424, '4-2-4', 'Ultra-attacking formation with four forwards.', NULL, NULL),
(@formation_id_4231, '4-2-3-1', 'Versatile formation with defensive midfield cover.', NULL, NULL),
(@formation_id_4411, '4-4-1-1', 'Balanced formation with a supporting second striker.', NULL, NULL),
(@formation_id_4222_narrow, '4-2-2-2 (Narrow)', 'Narrow formation with dual strikers and attacking mids.', NULL, NULL),
(@formation_id_532, '5-3-2', 'Defensive formation with wing-back support.', NULL, NULL),
(@formation_id_541, '5-4-1', 'Very defensive, counter-attacking formation.', NULL, NULL),
(@formation_id_451, '4-5-1', 'Midfield-heavy formation for control and defense.', NULL, NULL),
(@formation_id_4141, '4-1-4-1', 'Defensive with a single holding midfielder.', NULL, NULL),
(@formation_id_4222_brazil, '4-2-2-2 (Brazil)', 'Popular in Brazil with two defensive mids.', NULL, NULL),
(@formation_id_3223, '3-2-2-3', 'Total football formation with fluid positions.', NULL, NULL),
(@formation_id_3313, '3-3-1-3', 'Attacking formation with midfield diamond.', NULL, NULL),
(@formation_id_442d, '4-4-2 Diamond', 'Narrow midfield with diamond shape.', NULL, NULL);

-- Insert Event (global)
SET @event_id_league = UUID();
INSERT INTO events (id, name) VALUES
(@event_id_league, 'Premier League');

-- Insert Players for the New Team
SET @p_new_1 = UUID(); SET @p_new_2 = UUID(); SET @p_new_3 = UUID(); SET @p_new_4 = UUID();
SET @p_new_5 = UUID(); SET @p_new_6 = UUID(); SET @p_new_7 = UUID(); SET @p_new_8 = UUID();
SET @p_new_9 = UUID(); SET @p_new_10 = UUID(); SET @p_new_11 = UUID();
INSERT INTO players (id, team_id, name, position, jersey_number, birth_date, dominant_foot, height_cm, weight_kg, nationality, country_code, image_url, market_value) VALUES
(@p_new_1, @team_id_new, 'GK Alpha', 'GK', 1, '1995-05-10', 'right', 190, 85, 'Portuguese', 'PT', 'https://cdn.example.com/players/alpha.png', 15000000.00),
(@p_new_2, @team_id_new, 'LB Beta', 'DEF', 2, '1998-03-15', 'left', 178, 72, 'English', 'GB', 'https://cdn.example.com/players/beta.png', 12000000.00),
(@p_new_3, @team_id_new, 'CB Gamma', 'DEF', 3, '1993-07-20', 'right', 188, 83, 'Brazilian', 'BR', 'https://cdn.example.com/players/gamma.png', 20000000.00),
(@p_new_4, @team_id_new, 'CB Delta', 'DEF', 4, '1996-11-01', 'right', 185, 80, 'Dutch', 'NL', 'https://cdn.example.com/players/delta.png', 18000000.00),
(@p_new_5, @team_id_new, 'RB Epsilon', 'DEF', 5, '1997-09-25', 'right', 175, 70, 'Spanish', 'ES', 'https://cdn.example.com/players/epsilon.png', 14000000.00),
(@p_new_6, @team_id_new, 'DM Zeta', 'MID', 6, '1994-01-05', 'right', 180, 75, 'French', 'FR', 'https://cdn.example.com/players/zeta.png', 25000000.00),
(@p_new_7, @team_id_new, 'CM Eta', 'MID', 7, '1999-02-28', 'right', 176, 71, 'German', 'DE', 'https://cdn.example.com/players/eta.png', 30000000.00),
(@p_new_8, @team_id_new, 'AM Theta', 'MID', 8, '2000-04-12', 'left', 170, 68, 'Argentinian', 'AR', 'https://cdn.example.com/players/theta.png', 40000000.00),
(@p_new_9, @team_id_new, 'LW Iota', 'FWD', 9, '1992-06-30', 'right', 172, 69, 'Belgian', 'BE', 'https://cdn.example.com/players/iota.png', 22000000.00),
(@p_new_10, @team_id_new, 'ST Kappa', 'FWD', 10, '1994-08-18', 'right', 182, 78, 'Uruguayan', 'UY', 'https://cdn.example.com/players/kappa.png', 28000000.00),
(@p_new_11, @team_id_new, 'RW Lambda', 'FWD', 11, '1996-10-05', 'left', 174, 70, 'Italian', 'IT', 'https://cdn.example.com/players/lambda.png', 27000000.00);

-- Insert a Dummy Opponent Team (not linked to any user)
SET @team_id_opponent = UUID();
INSERT INTO teams (id, name, primary_color, secondary_color, logo_url) VALUES
(@team_id_opponent, 'Opponent FC', '#FF0000', '#000000', 'https://cdn.example.com/teams/opponent_fc.png');

-- Insert a Match for the New Team
SET @match_id_new = UUID();
INSERT INTO matches (id, home_team_id, away_team_id, date_time, venue, status, home_score, away_score, event_id) VALUES
(@match_id_new, @team_id_new, @team_id_opponent, '2024-08-01 18:30:00', 'Home Ground', 'completed', 3, 1, @event_id_league);

-- Insert Match Lineups (explicit id provided)
INSERT INTO match_lineups (id, match_id, team_id, formation_id, is_starting, player_id, position_in_formation) VALUES
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_1, '0.5,0.95'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_2, '0.1,0.75'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_3, '0.3,0.8'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_4, '0.7,0.8'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_5, '0.9,0.75'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_6, '0.5,0.6'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_7, '0.3,0.5'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_8, '0.7,0.5'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_9, '0.1,0.2'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_10, '0.5,0.1'),
(UUID(), @match_id_new, @team_id_new, @formation_id_433, TRUE, @p_new_11, '0.9,0.2');

-- Insert Player Match Statistics for the New Team's players (added id per row)
INSERT INTO player_match_statistics (id, match_id, player_id, minutes_played, shots, shots_on_target, passes, accurate_passes, tackles, key_passes, player_xg, progressive_carries, press_resistance_success_rate, defensive_coverage_km, notes, rating) VALUES
(UUID(), @match_id_new, @p_new_1, 90, 0, 0, 20, 18, 0, 0, 0.0, 0, 100.0, 0.5, 'Solid goalkeeping, few saves needed.', 7.0),
(UUID(), @match_id_new, @p_new_10, 90, 5, 3, 25, 20, 1, 2, 1.2, 8, 90.0, 3.0, 'Scored two goals, clinical finishing.', 9.5);

-- Insert Team Match Statistics for the New Team (id added)
INSERT INTO match_team_statistics (id, match_id, team_id, possession_percentage, total_shots, shots_on_target, expected_goals, pressures, final_third_passes, build_up_patterns, defensive_block_patterns, high_turnover_zones_data, set_piece_xg_breakdown_data, transition_speed_data) VALUES
(UUID(), @match_id_new, @team_id_new, 60.0, 15, 8, 2.5, 120, 70,
    '{"left_side_progression_percent": 65, "most_frequent_pass_chain": ["CB Gamma", "CM Eta", "ST Kappa"]}',
    '{"out_of_possession_formation": "4-4-2", "duel_success_in_box_percent": 70}',
    '{"zone1": 6, "zone2": 4}',
    '{"corners": 0.6, "free_kicks": 0.4}',
    '{"attack_to_defense_sec": 3.0, "defense_to_attack_sec": 2.5}');

-- Insert Players for the Opponent Team
SET @opp_p_1 = UUID(); SET @opp_p_2 = UUID(); SET @opp_p_3 = UUID(); SET @opp_p_4 = UUID();
SET @opp_p_5 = UUID(); SET @opp_p_6 = UUID(); SET @opp_p_7 = UUID(); SET @opp_p_8 = UUID();
SET @opp_p_9 = UUID(); SET @opp_p_10 = UUID(); SET @opp_p_11 = UUID();

INSERT INTO players (id, team_id, name, position, jersey_number, birth_date, dominant_foot, height_cm, weight_kg, nationality, country_code, image_url, market_value) VALUES
(@opp_p_1, @team_id_opponent, 'Opponent GK', 'GK', 1, '1996-01-20', 'right', 188, 84, 'German', 'DE', 'https://cdn.example.com/players/opp_gk.png', 10000000.00),
(@opp_p_2, @team_id_opponent, 'Opponent LB', 'DEF', 2, '1997-02-25', 'left', 175, 70, 'French', 'FR', 'https://cdn.example.com/players/opp_lb.png', 8000000.00),
(@opp_p_3, @team_id_opponent, 'Opponent CB1', 'DEF', 3, '1994-04-01', 'right', 186, 82, 'Spanish', 'ES', 'https://cdn.example.com/players/opp_cb1.png', 15000000.00),
(@opp_p_4, @team_id_opponent, 'Opponent CB2', 'DEF', 4, '1995-06-10', 'right', 184, 80, 'Italian', 'IT', 'https://cdn.example.com/players/opp_cb2.png', 14000000.00),
(@opp_p_5, @team_id_opponent, 'Opponent RB', 'DEF', 5, '1998-08-15', 'right', 176, 71, 'Dutch', 'NL', 'https://cdn.example.com/players/opp_rb.png', 9000000.00),
(@opp_p_6, @team_id_opponent, 'Opponent CM1', 'MID', 6, '1993-09-20', 'right', 179, 74, 'Portuguese', 'PT', 'https://cdn.example.com/players/opp_cm1.png', 18000000.00),
(@opp_p_7, @team_id_opponent, 'Opponent CM2', 'MID', 7, '1999-11-05', 'left', 177, 72, 'English', 'GB', 'https://cdn.example.com/players/opp_cm2.png', 20000000.00),
(@opp_p_8, @team_id_opponent, 'Opponent LW', 'FWD', 8, '2000-01-12', 'right', 170, 67, 'Brazilian', 'BR', 'https://cdn.example.com/players/opp_lw.png', 25000000.00),
(@opp_p_9, @team_id_opponent, 'Opponent RW', 'FWD', 9, '1992-03-28', 'left', 173, 68, 'Argentinian', 'AR', 'https://cdn.example.com/players/opp_rw.png', 15000000.00),
(@opp_p_10, @team_id_opponent, 'Opponent ST1', 'FWD', 10, '1994-05-18', 'right', 180, 76, 'Uruguayan', 'UY', 'https://cdn.example.com/players/opp_st1.png', 22000000.00),
(@opp_p_11, @team_id_opponent, 'Opponent ST2', 'FWD', 11, '1996-07-05', 'right', 178, 74, 'Belgian', 'BE', 'https://cdn.example.com/players/opp_st2.png', 20000000.00);

-- Insert Match Lineups for the Opponent Team (Away Team) with explicit ids
INSERT INTO match_lineups (id, match_id, team_id, formation_id, is_starting, player_id, position_in_formation) VALUES
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_1, '0.5,0.05'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_2, '0.1,0.25'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_3, '0.3,0.2'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_4, '0.7,0.2'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_5, '0.9,0.25'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_6, '0.1,0.5'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_7, '0.3,0.5'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_8, '0.7,0.5'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_9, '0.9,0.5'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_10, '0.3,0.8'),
(UUID(), @match_id_new, @team_id_opponent, @formation_id_442, TRUE, @opp_p_11, '0.7,0.8');

-- Insert Player Match Statistics for remaining players of 'My Awesome Team' (ids added)
INSERT INTO player_match_statistics (id, match_id, player_id, minutes_played, shots, shots_on_target, passes, accurate_passes, tackles, interceptions, clearances, saves, fouls_committed, fouls_suffered, offsides, distance_covered_km, player_xg, key_passes, progressive_carries, press_resistance_success_rate, defensive_coverage_km, notes, rating) VALUES
(UUID(), @match_id_new, @p_new_2, 90, 0, 0, 45, 40, 3, 2, 1, 0, 1, 1, 0, 9.8, 0.0, 0, 0, 85.0, 4.5, 'Solid defensive performance.', 7.2),
(UUID(), @match_id_new, @p_new_3, 90, 1, 0, 50, 48, 4, 3, 2, 0, 0, 1, 0, 10.5, 0.1, 0, 0, 90.0, 5.0, 'Strong in aerial duels.', 7.5),
(UUID(), @match_id_new, @p_new_4, 90, 0, 0, 55, 52, 3, 2, 2, 0, 1, 0, 0, 10.0, 0.0, 0, 0, 88.0, 4.8, 'Good positioning and interceptions.', 7.3),
(UUID(), @match_id_new, @p_new_5, 90, 1, 0, 48, 42, 2, 1, 0, 0, 1, 2, 0, 9.5, 0.1, 1, 2, 75.0, 4.0, 'Active on the right flank.', 7.0),
(UUID(), @match_id_new, @p_new_6, 90, 1, 1, 60, 55, 5, 4, 1, 0, 2, 3, 0, 11.0, 0.2, 2, 3, 92.0, 6.0, 'Controlled the midfield.', 8.0),
(UUID(), @match_id_new, @p_new_7, 90, 2, 1, 65, 60, 2, 1, 0, 0, 1, 1, 0, 10.8, 0.3, 3, 4, 80.0, 5.5, 'Key passes and good link-up play.', 8.2),
(UUID(), @match_id_new, @p_new_8, 90, 3, 2, 50, 45, 1, 0, 0, 0, 0, 1, 0, 9.0, 0.5, 4, 6, 70.0, 3.0, 'Creative force, assisted a goal.', 8.5),
(UUID(), @match_id_new, @p_new_9, 90, 4, 2, 30, 25, 0, 0, 0, 0, 1, 2, 1, 8.5, 0.7, 2, 5, 60.0, 2.5, 'Dangerous on the left wing.', 7.8),
(UUID(), @match_id_new, @p_new_11, 90, 3, 1, 32, 28, 0, 0, 0, 0, 0, 1, 0, 8.8, 0.6, 1, 4, 65.0, 2.8, 'Good crosses and runs.', 7.6);

-- Opponent FC players statistics (ids added)
INSERT INTO player_match_statistics (id, match_id, player_id, minutes_played, shots, shots_on_target, passes, accurate_passes, tackles, interceptions, clearances, saves, fouls_committed, fouls_suffered, offsides, distance_covered_km, player_xg, key_passes, progressive_carries, press_resistance_success_rate, defensive_coverage_km, notes, rating) VALUES
(UUID(), @match_id_new, @opp_p_1, 90, 0, 0, 22, 20, 0, 0, 0, 6, 0, 0, 0, 7.5, 0.0, 0, 0, 100.0, 0.6, 'Made some good saves.', 7.0),
(UUID(), @match_id_new, @opp_p_2, 90, 0, 0, 38, 35, 2, 1, 1, 0, 1, 1, 0, 9.0, 0.0, 0, 0, 80.0, 3.8, 'Struggled with pace on the wing.', 6.5),
(UUID(), @match_id_new, @opp_p_3, 90, 0, 0, 45, 40, 3, 2, 3, 0, 2, 1, 0, 9.5, 0.0, 0, 0, 85.0, 4.2, 'Solid in defense.', 6.8),
(UUID(), @match_id_new, @opp_p_4, 90, 0, 0, 42, 38, 3, 1, 4, 0, 1, 0, 0, 9.2, 0.0, 0, 0, 82.0, 4.0, 'Good clearances.', 6.7),
(UUID(), @match_id_new, @opp_p_5, 90, 0, 0, 35, 30, 2, 1, 1, 0, 1, 2, 0, 8.8, 0.0, 0, 0, 78.0, 3.5, 'Active on the right.', 6.6),
(UUID(), @match_id_new, @opp_p_6, 90, 1, 0, 50, 45, 4, 3, 0, 0, 2, 2, 0, 10.0, 0.1, 1, 1, 88.0, 5.0, 'Worked hard in midfield.', 7.0),
(UUID(), @match_id_new, @opp_p_7, 90, 1, 1, 48, 42, 3, 2, 0, 0, 1, 1, 0, 9.8, 0.2, 1, 2, 80.0, 4.8, 'Tried to create chances.', 7.1),
(UUID(), @match_id_new, @opp_p_8, 90, 2, 1, 25, 20, 0, 0, 0, 0, 1, 1, 1, 7.0, 0.4, 1, 3, 60.0, 2.0, 'Scored the only goal.', 7.5),
(UUID(), @match_id_new, @opp_p_9, 90, 1, 0, 28, 23, 0, 0, 0, 0, 0, 1, 0, 7.2, 0.3, 0, 2, 55.0, 1.8, 'Limited impact.', 6.0),
(UUID(), @match_id_new, @opp_p_10, 90, 3, 1, 20, 15, 0, 0, 0, 0, 1, 2, 1, 7.8, 0.6, 1, 1, 50.0, 1.5, 'Isolated upfront.', 6.2),
(UUID(), @match_id_new, @opp_p_11, 90, 2, 0, 18, 12, 0, 0, 0, 0, 0, 1, 0, 7.5, 0.5, 0, 1, 45.0, 1.2, 'Struggled to get involved.', 5.8);

-- Insert Team Match Statistics for the Opponent Team (id added)
INSERT INTO match_team_statistics (id, match_id, team_id, possession_percentage, total_shots, shots_on_target, expected_goals, pressures, final_third_passes, high_turnover_zones_data, set_piece_xg_breakdown_data, transition_speed_data, build_up_patterns, defensive_block_patterns) VALUES
(UUID(), @match_id_new, @team_id_opponent, 39.0, 10, 4, 1.0, 100, 50,
    '{"zone1": 3, "zone2": 2}',
    '{"corners": 0.3, "free_kicks": 0.2}',
    '{"attack_to_defense_sec": 4.0, "defense_to_attack_sec": 3.5}',
    '{"long_ball_percent": 70, "counter_attack_success_rate": 40}',
    '{"defensive_line": "deep", "pressing_intensity": "low"}');

-- Insert Match Events (ids added)
INSERT INTO match_events (id, match_id, player_id, event_type, minute, video_timestamp, coordinates) VALUES
(UUID(), @match_id_new, @p_new_10, 'goal', 15, 900.0, '0.8,0.7'),
(UUID(), @match_id_new, @p_new_8, 'assist', 15, 899.0, '0.7,0.6'),
(UUID(), @match_id_new, @opp_p_8, 'goal', 30, 1800.0, '0.2,0.3'),
(UUID(), @match_id_new, @p_new_6, 'yellow_card', 40, 2400.0, '0.5,0.5'),
(UUID(), @match_id_new, @p_new_10, 'goal', 55, 3300.0, '0.75,0.75'),
(UUID(), @match_id_new, @p_new_7, 'assist', 55, 3299.0, '0.6,0.7'),
(UUID(), @match_id_new, @p_new_9, 'sub_out', 70, 4200.0, NULL),
(UUID(), @match_id_new, @p_new_11, 'sub_in', 70, 4200.0, NULL),
(UUID(), @match_id_new, @p_new_3, 'red_card', 85, 5100.0, '0.3,0.8');

-- Insert Staff Members for the New Team
SET @staff_id_1 = UUID();
SET @staff_id_2 = UUID();
INSERT INTO staff (id, team_id, user_id, name, role, permission_level, email) VALUES
(@staff_id_1, @team_id_new, NULL, 'Adem Coach', 'head_coach', 'full_access', 'adem@example.com'),
(@staff_id_2, @team_id_new, NULL, 'John Analyst', 'analyst', 'notes_only', 'john@example.com');

-- Insert Match Notes for the existing match
INSERT INTO match_notes (id, match_id, user_id, content, note_type, video_timestamp) VALUES
(UUID(), @match_id_new, @user_id_new, 'High press from the beginning. Focus on their slow CBs.', 'pre_match', 0.0),
(UUID(), @match_id_new, @user_id_new, 'Strikers are staying too wide. Need to tuck in more.', 'live_reaction', 320.5),
(UUID(), @match_id_new, @user_id_new, 'Counter-press success rate is high. Keep exploiting it.', 'tactical', 1250.0),
(UUID(), @match_id_new, @user_id_new, 'Formation shift to 4-4-2 worked well for stability.', 'tactical', 5400.0);

-- Insert Reunions
INSERT INTO reunions (id, title, date, location, icon_name) VALUES
(UUID(), 'Tactical Briefing', '2024-07-30 14:00:00', 'Video Room 1', 'psychology'),
(UUID(), 'Team Dinner', '2024-08-01 21:00:00', 'The Grand Tavern', 'restaurant'),
(UUID(), 'Pre-Season Logistics', '2024-07-25 10:00:00', 'Boardroom', 'settings');

-- Insert Training Sessions
INSERT INTO training_sessions (id, title, date, focus, icon_name) VALUES
(UUID(), 'Defensive Drills', '2024-07-28 09:00:00', 'Zonal Marking', 'shield'),
(UUID(), 'Precision Shooting', '2024-07-29 11:30:00', 'Finishing', 'sports_soccer'),
(UUID(), 'Midfield Transition', '2024-07-27 15:00:00', 'Counter-Attack', 'trending_flat');

-- Insert Analysis Reports
SET @report_id_1 = UUID();
INSERT INTO analysis_reports (id, match_id, report_type, report_data, generated_by) VALUES
(@report_id_1, @match_id_new, 'Tactical Post-Match', '{"summary": "Dominant performance with a high line.", "strengths": ["Midfield control", "Finishing"], "weaknesses": ["Defensive transition"]}', @user_id_new);

-- Insert Video Segments
INSERT INTO video_segments (id, match_id, event_id, analysis_report_id, start_time_sec, end_time_sec, description, video_url) VALUES
(UUID(), @match_id_new, NULL, @report_id_1, 900.0, 915.0, 'Goal 1 build-up', 'https://video.example.com/match1/goal1.mp4'),
(UUID(), @match_id_new, NULL, @report_id_1, 3300.0, 3310.0, 'Goal 2 finishing touch', 'https://video.example.com/match1/goal2.mp4');

-- End of full_insert.sql
