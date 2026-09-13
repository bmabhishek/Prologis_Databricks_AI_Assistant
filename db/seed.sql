-- Seed: 20 Prologis-style industrial/logistics properties across major US metros
-- Run: psql -U postgres -d financial_assistant -f seed.sql

INSERT INTO properties (address, metro_area, sq_footage, property_type) VALUES
('1545 W 190th St, Torrance', 'Los Angeles',  450000, 'Industrial'),
('2200 E Carson St, Long Beach', 'Los Angeles', 380000, 'Industrial'),
('5400 W Roosevelt Rd, Cicero', 'Chicago',     620000, 'Industrial'),
('1701 Cornell Ave, Melrose Park', 'Chicago',  295000, 'Logistics'),
('8800 Bryn Mawr Ave, Chicago', 'Chicago',     510000, 'Industrial'),
('3500 Bayshore Rd, Edison', 'New York',       415000, 'Logistics'),
('200 Middlesex Ave, Carteret', 'New York',    540000, 'Industrial'),
('1600 Westport Rd, Kansas City', 'Kansas City', 280000, 'Warehouse'),
('900 N Westmoreland Rd, Dallas', 'Dallas',    635000, 'Industrial'),
('4400 S Westmoreland Rd, Dallas', 'Dallas',   470000, 'Logistics'),
('14600 Trinity Blvd, Fort Worth', 'Dallas',   390000, 'Warehouse'),
('8500 NW 25th St, Doral', 'Miami',            325000, 'Industrial'),
('11500 NW 100th Rd, Medley', 'Miami',         410000, 'Logistics'),
('22500 76th Ave S, Kent', 'Seattle',          355000, 'Industrial'),
('1900 W Valley Hwy N, Auburn', 'Seattle',     285000, 'Warehouse'),
('5500 W Buckeye Rd, Phoenix', 'Phoenix',      575000, 'Industrial'),
('7400 W Roosevelt St, Phoenix', 'Phoenix',    340000, 'Logistics'),
('3300 NW Aloclek Dr, Hillsboro', 'Portland',  265000, 'Warehouse'),
('44 Twosome Dr, Moorestown', 'Philadelphia',  445000, 'Industrial'),
('500 Mansell Rd, Roswell', 'Atlanta',         390000, 'Logistics');

-- Financials (revenue, net_income, expenses) — values in USD, FY2024
-- Roughly proportional to sq footage with some noise
INSERT INTO financials (property_id, revenue, net_income, expenses, fiscal_year) VALUES
(1,  8100000,  2430000,  5670000, 2024),
(2,  6840000,  2052000,  4788000, 2024),
(3, 11160000,  3348000,  7812000, 2024),
(4,  5310000,  1593000,  3717000, 2024),
(5,  9180000,  2754000,  6426000, 2024),
(6,  7470000,  2241000,  5229000, 2024),
(7,  9720000,  2916000,  6804000, 2024),
(8,  5040000,  1512000,  3528000, 2024),
(9, 11430000,  3429000,  8001000, 2024),
(10, 8460000,  2538000,  5922000, 2024),
(11, 7020000,  2106000,  4914000, 2024),
(12, 5850000,  1755000,  4095000, 2024),
(13, 7380000,  2214000,  5166000, 2024),
(14, 6390000,  1917000,  4473000, 2024),
(15, 5130000,  1539000,  3591000, 2024),
(16, 10350000, 3105000,  7245000, 2024),
(17, 6120000,  1836000,  4284000, 2024),
(18, 4770000,  1431000,  3339000, 2024),
(19, 8010000,  2403000,  5607000, 2024),
(20, 7020000,  2106000,  4914000, 2024);
