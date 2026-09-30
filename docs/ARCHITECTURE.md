# Architecture: [Product name]

<!--
The map of how the app is built and how its parts connect.
The agent can read your files. It cannot read the decisions behind them.
Write those down here. When the structure changes, update this file in
the same commit.
-->

## 01 System overview

One level above the code. Boxes and arrows, not modules.

```
Client (browser / mobile)
    |
    v
Frontend (app + pages)  ----->  Auth provider
    |
    v
Backend API (route handlers / server actions)
    |
    +--->  Database
    +--->  Payments
    +--->  Email
    +--->  File storage
```

- Users only ever interact with the frontend.
- The frontend never talks to the database directly. Everything goes through the API layer.
- Third-party services are called from the backend only. Keys never reach the client.

## 02 Tech stack

| Layer | Choice | Why this, not the obvious alternative |
| --- | --- | --- |
| Frontend | [Next.js, React, TypeScript] | [...] |
| Styling | [Tailwind, tokens from DESIGN_SYSTEM.md] | [...] |
| Backend | [Next.js route handlers] | [...] |
| Database | [Postgres] | [...] |
| Queries | [Drizzle / Prisma / SQL] | [...] |
| Auth | [Provider] | [...] |
| Payments | [Stripe] | [...] |
| Email | [Provider] | [...] |
| Hosting | [Vercel] | [...] |
| Testing | [Vitest + Playwright] | [...] |

Do not add a dependency that overlaps with a row in this table without asking.

## 03 Project structure

```
/
  AGENTS.md            agent instructions (read first)
  docs/                PRD, design system, this file
  src/
    app/               routes and pages only. No business logic here.
    components/
      ui/              primitives: Button, Input, Card (see DESIGN_SYSTEM.md)
      features/        feature-level components, one folder per feature
    lib/
      db/              schema, migrations, queries
      services/        business logic, one file per domain (billing.ts, auth.ts)
      integrations/    third-party clients (stripe.ts, email.ts)
      utils/           pure helpers, no side effects
    types/             shared TypeScript types
  public/              static assets
  tests/               end-to-end tests
```

Where new code belongs:
- A new page: src/app/<route>/page.tsx. Thin. Calls a service.
- New business logic: src/lib/services/<domain>.ts.
- A new third-party call: src/lib/integrations/<vendor>.ts, imported by a service, never by a component.
- A new UI primitive: src/components/ui/, added to DESIGN_SYSTEM.md in the same commit.

## 04 Data flow

Request path for every write:
1. User action in a component
2. Route handler validates the input (schema first, then logic)
3. Service function applies the business rule
4. Query layer touches the database
5. Typed response returns. The UI updates from the response, not from a guess.

State:
- Server state lives in the database and is fetched per request.
- Client state is UI only (open menus, form drafts). No business data in global client state.
- Optimistic updates allowed for: [list them, or "none in v1"]

Auth:
- The session is checked at the route level. Services receive a user id and never read the session themselves.

Errors:
- Validation errors return to the form with a message.
- Unexpected errors are logged with a request id and shown as a generic message.
  Never show stack traces or vendor error text to the user.

## 05 Boundaries

Dependencies flow one way: app -> components -> services -> db / integrations.

Allowed:
- Pages import components and services.
- Services import db and integrations.

Never:
- Components import from db or integrations directly.
- Services import from components or app.
- Anything in the client bundle reads a secret. Server-only modules stay server-only.

## 06 Decisions that look wrong but are intentional

The agent must not "fix" these.

- [Example: the order row stores its own price instead of joining to products,
  so old orders survive price changes.]
- [Example: the auth check is repeated in every route on purpose.
  Global middleware was tried, it hid failures.]
- [Add yours.]

## 07 Scalability and future considerations

- Expected scale for v1: [users, requests per day, data size]. Build for this, not for 100x.
- Known limits: [e.g. single region, no job queue, cron every 5 minutes].
- Planned next: [e.g. background queue once email volume passes X].
- Not planned: [e.g. multi-tenant, native mobile]. Do not add abstractions for these.

## 08 When the agent must stop and ask

If a task needs any of the following, stop, name the conflict, and propose
the smallest change that avoids it:
- Crossing a boundary in section 05
- Changing a decision in section 06
- Adding a table, a dependency, or a third-party service
- Changing the auth or payment flow
