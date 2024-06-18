WITH RECURSIVE subordinates AS (
    SELECT
        dpf.code,
        dpf.code AS categoryCode,
        dpf.code AS originalFamilyId,
        'root' AS code_parent, -- Set 'root' as the parent for the first level
        'root' AS parent_categories, -- Set 'root' as the parent category for the first level
        1 AS depth -- Start depth at 1 for the base case
    FROM
        `c:/Users/tranb/Documents/test/data.csv` dpf
    WHERE dpf.code_parent IS NULL
    UNION ALL
        SELECT
            e.code,
            e.code AS categoryCode,
            s.originalFamilyId,
            COALESCE(e.code_parent, 'root') AS code_parent,
            COALESCE(e.parent_categories, 'root') AS parent_categories,
            s.depth + 1 AS depth
        FROM
            `c:/Users/tranb/Documents/test/data.csv` e
        JOIN subordinates s ON e.code_parent = s.code
)
, max_depth AS (
    SELECT MAX(depth) AS max_depth
    FROM subordinates
)
SELECT
    s.code AS OriginalCode,
    s.originalFamilyId AS OriginalFamilyId,
    s.categoryCode AS categoryCode,
    s.originalFamilyId AS categoryFamilyId,
    s.code_parent AS code_parent,
    s.parent_categories AS parent_categories,
    (SELECT max_depth.max_depth FROM max_depth) - s.depth + 1 AS Level, -- Correctly access the max_depth column from the struct
    CURRENT_TIMESTAMP AS UpdateDate
FROM
    subordinates s
WHERE s.code = 'porte_interieure';
