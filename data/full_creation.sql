-- full_creation.sql
-- Schema and trigger creation for soccer_analytics

CREATE DATABASE IF NOT EXISTS soccer_analytics;
USE soccer_analytics;

-- Drop existing objects if they exist (for easy re-execution)
DROP TABLE IF EXISTS match_notes, match_team_statistics, match_lineups, player_match_statistics, video_segments, match_events, analysis_reports, staff, players, matches, teams, users, formations, reunions, training_sessions, events;

-- Independent Tables
CREATE TABLE users (
    id CHAR(36) PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    user_type ENUM('owner', 'staff') DEFAULT 'owner',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL
);

CREATE TABLE teams (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    user_id CHAR(36),
    primary_color CHAR(7) DEFAULT '#0000FF',
    secondary_color CHAR(7),
    logo_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE formations (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    positions JSON,
    user_id CHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE reunions (
    id CHAR(36) PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    date DATETIME NOT NULL,
    location VARCHAR(100),
    icon_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE training_sessions (
    id CHAR(36) PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    date DATETIME NOT NULL,
    focus VARCHAR(100),
    icon_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE events (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dependent Tables
CREATE TABLE players (
    id CHAR(36) PRIMARY KEY,
    team_id CHAR(36),
    name VARCHAR(50) NOT NULL,
    position ENUM('GK', 'DEF', 'MID', 'FWD'),
    jersey_number TINYINT UNSIGNED,
    birth_date DATE,
    dominant_foot ENUM('left', 'right'),
    height_cm SMALLINT UNSIGNED,
    weight_kg SMALLINT UNSIGNED,
    nationality VARCHAR(50),
    country_code CHAR(2),
    image_url VARCHAR(255) NULL,
    market_value DECIMAL(10,2),
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
);

CREATE TABLE staff (
    id CHAR(36) PRIMARY KEY,
    team_id CHAR(36),
    user_id CHAR(36) NULL,
    name VARCHAR(50) NOT NULL,
    role ENUM('head_coach', 'assistant_coach', 'physio', 'analyst'),
    permission_level ENUM('full_access', 'view_only', 'notes_only') DEFAULT 'view_only',
    email VARCHAR(100) NULL,
    UNIQUE KEY (user_id),
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE matches (
    id CHAR(36) PRIMARY KEY,
    home_team_id CHAR(36) NOT NULL,
    away_team_id CHAR(36) NOT NULL,
    date_time DATETIME NOT NULL,
    venue VARCHAR(100),
    status ENUM('upcoming', 'live', 'completed'),
    home_score TINYINT UNSIGNED DEFAULT 0,
    away_score TINYINT UNSIGNED DEFAULT 0,
    event_id CHAR(36),
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE match_events (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36),
    player_id CHAR(36),
    event_type ENUM('goal', 'assist', 'yellow_card', 'red_card', 'sub_in', 'sub_out'),
    minute SMALLINT UNSIGNED NOT NULL,
    video_timestamp FLOAT,
    coordinates VARCHAR(50),
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE TABLE player_match_statistics (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    player_id CHAR(36) NOT NULL,
    minutes_played SMALLINT UNSIGNED,
    shots SMALLINT UNSIGNED DEFAULT 0,
    shots_on_target SMALLINT UNSIGNED DEFAULT 0,
    passes SMALLINT UNSIGNED DEFAULT 0,
    accurate_passes SMALLINT UNSIGNED DEFAULT 0,
    tackles SMALLINT UNSIGNED DEFAULT 0,
    interceptions SMALLINT UNSIGNED DEFAULT 0,
    clearances SMALLINT UNSIGNED DEFAULT 0,
    saves SMALLINT UNSIGNED DEFAULT 0,
    fouls_committed SMALLINT UNSIGNED DEFAULT 0,
    fouls_suffered SMALLINT UNSIGNED DEFAULT 0,
    offsides SMALLINT UNSIGNED DEFAULT 0,
    distance_covered_km DECIMAL(5,2),
    player_xg DECIMAL(5,2) DEFAULT 0.0,
    key_passes INT DEFAULT 0,
    progressive_carries INT DEFAULT 0,
    press_resistance_success_rate DECIMAL(5,2) DEFAULT 0.0,
    defensive_coverage_km DECIMAL(5,2) DEFAULT 0.0,
    notes TEXT,
    rating DECIMAL(3,1),
    UNIQUE KEY (match_id, player_id),
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
);

CREATE TABLE analysis_reports (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    report_type VARCHAR(50),
    report_data JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_by CHAR(36),
    UNIQUE KEY (match_id), -- One report per match for simplicity in this version
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (generated_by) REFERENCES users(id)
);

CREATE TABLE video_segments (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    event_id CHAR(36),
    analysis_report_id CHAR(36),
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT NOT NULL,
    description VARCHAR(255),
    video_url VARCHAR(255),
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES match_events(id) ON DELETE SET NULL,
    FOREIGN KEY (analysis_report_id) REFERENCES analysis_reports(id) ON DELETE SET NULL
);

CREATE TABLE match_lineups (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    team_id CHAR(36) NOT NULL,
    formation_id CHAR(36),
    is_starting BOOLEAN NOT NULL,
    player_id CHAR(36) NOT NULL,
    position_in_formation VARCHAR(50),
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (formation_id) REFERENCES formations(id) ON DELETE SET NULL,
    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
);

CREATE TABLE match_team_statistics (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    team_id CHAR(36) NOT NULL,
    possession_percentage DECIMAL(5,2) DEFAULT 0.0,
    total_shots INT DEFAULT 0,
    shots_on_target INT DEFAULT 0,
    expected_goals DECIMAL(5,2) DEFAULT 0.0,
    pressures INT DEFAULT 0,
    final_third_passes INT DEFAULT 0,
    high_turnover_zones_data JSON,
    set_piece_xg_breakdown_data JSON,
    transition_speed_data JSON,
    build_up_patterns JSON,
    defensive_block_patterns JSON,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
);

CREATE TABLE match_notes (
    id CHAR(36) PRIMARY KEY,
    match_id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    content TEXT NOT NULL,
    note_type ENUM('pre_match', 'live_reaction', 'tactical') DEFAULT 'tactical',
    video_timestamp FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- UUID Triggers (add after all tables exist)
DELIMITER //
CREATE TRIGGER before_users_insert BEFORE INSERT ON users FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_teams_insert BEFORE INSERT ON teams FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_players_insert BEFORE INSERT ON players FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_staff_insert BEFORE INSERT ON staff FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_matches_insert BEFORE INSERT ON matches FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_match_events_insert BEFORE INSERT ON match_events FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_player_match_statistics_insert BEFORE INSERT ON player_match_statistics FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_analysis_reports_insert BEFORE INSERT ON analysis_reports FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_video_segments_insert BEFORE INSERT ON video_segments FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_formations_insert BEFORE INSERT ON formations FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_match_lineups_insert BEFORE INSERT ON match_lineups FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_reunions_insert BEFORE INSERT ON reunions FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_training_sessions_insert BEFORE INSERT ON training_sessions FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_events_insert BEFORE INSERT ON events FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_match_team_statistics_insert BEFORE INSERT ON match_team_statistics FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
CREATE TRIGGER before_match_notes_insert BEFORE INSERT ON match_notes FOR EACH ROW
BEGIN IF NEW.id IS NULL THEN SET NEW.id = UUID(); END IF; END//
DELIMITER ;

-- End of schema & triggers
