-- Financial Assistant DB Schema
-- Run: psql -U postgres -d financial_assistant -f schema.sql

DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS properties;

CREATE TABLE properties (
    property_id    SERIAL PRIMARY KEY,
    address        VARCHAR(255) NOT NULL,
    metro_area     VARCHAR(100) NOT NULL,
    sq_footage     INTEGER NOT NULL,
    property_type  VARCHAR(50) NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE financials (
    fin_id         SERIAL PRIMARY KEY,
    property_id    INTEGER REFERENCES properties(property_id) ON DELETE CASCADE,
    revenue        NUMERIC(14, 2) NOT NULL,
    net_income     NUMERIC(14, 2) NOT NULL,
    expenses       NUMERIC(14, 2) NOT NULL,
    fiscal_year    INTEGER DEFAULT 2024,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_properties_metro ON properties(metro_area);
CREATE INDEX idx_properties_type  ON properties(property_type);
CREATE INDEX idx_financials_pid   ON financials(property_id);
