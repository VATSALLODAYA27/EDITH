# Project Atlas

## Overview
Project Atlas is Nimbus Labs' internal data platform. It replaces the old nightly CSV exports with a single place where every team can query company data. It is owned by the Data team.

## Timeline
- Phase 1 (ingestion pipelines): completed November 2026.
- Phase 2 (self-service dashboards): in progress, due January 2027.
- Public launch to all employees: March 2027.

## Tech stack
Atlas uses PostgreSQL as the warehouse, Airbyte for ingestion, dbt for transformations, and Metabase for dashboards. Access is granted per team through the #atlas-access Slack channel.
