SET search_path TO public;

CREATE TABLE IF NOT EXISTS dim_protocol (
  protocol_id SERIAL PRIMARY KEY,
  protocol_name VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_service (
  service_id SERIAL PRIMARY KEY,
  service_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_flag (
  flag_id SERIAL PRIMARY KEY,
  flag_name VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_attack (
  attack_id SERIAL PRIMARY KEY,
  attack_name VARCHAR(50) UNIQUE NOT NULL,
  attack_category VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_network_traffic (
  traffic_id SERIAL PRIMARY KEY,
  protocol_id INTEGER REFERENCES dim_protocol(protocol_id),
  service_id INTEGER REFERENCES dim_service(service_id),
  flag_id INTEGER REFERENCES dim_flag(flag_id),
  attack_id INTEGER REFERENCES dim_attack(attack_id),

  duration INTEGER,
  src_bytes BIGINT,
  dst_bytes BIGINT,
  land INTEGER,
  wrong_fragment INTEGER,
  urgent INTEGER,
  count INTEGER,
  srv_count INTEGER,
  serror_rate FLOAT,
  srv_serror_rate FLOAT,
  rerror_rate FLOAT,
  srv_rerror_rate FLOAT,
  same_srv_rate FLOAT,
  diff_srv_rate FLOAT,
  srv_diff_host_rate FLOAT,
  dst_host_count INTEGER,
  dst_host_srv_count INTEGER,
  dst_host_same_srv_rate FLOAT,
  dst_host_diff_srv_rate FLOAT,
  dst_host_same_src_port_rate FLOAT,
  dst_host_srv_diff_host_rate FLOAT,
  dst_host_serror_rate FLOAT,
  dst_host_srv_serror_rate FLOAT,
  dst_host_rerror_rate FLOAT,
  dst_host_srv_rerror_rate FLOAT,
  is_attack BOOLEAN,
  is_test_data BOOLEAN
);