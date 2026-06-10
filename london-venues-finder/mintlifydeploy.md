# Mintlify Deploy Guide

This file is a working checklist for the agent to follow when deploying this project with Mintlify. Paste the official Mintlify docs below when ready, then update the steps with any exact commands, required files, or settings from those docs.

## Docs To Paste

Paste the Mintlify deployment documentation here:

```md
<!-- Mintlify docs go here -->
```

## Deployment Goal

Deploy the London venues finder project using Mintlify, making sure the documentation site is configured correctly and can be previewed before publishing.

## Agent Checklist

1> ## Documentation Index
> Fetch the complete documentation index at: https://www.mintlify.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Trigger deployment

> Queue a deployment update for your documentation project. Returns a status ID that can be used to track the update progress. The update is triggered from your configured deployment branch.



## OpenAPI

````yaml /openapi.json POST /project/update/{projectId}
openapi: 3.0.1
info:
  title: Mintlify External API
  description: An API for Mintlify documentation management and resource access.
  version: 1.0.0
servers:
  - url: https://api.mintlify.com/v1
security:
  - bearerAuth: []
paths:
  /project/update/{projectId}:
    post:
      summary: Trigger update
      description: >-
        Queue a deployment update for your documentation project. Returns a
        status ID that can be used to track the update progress. The update is
        triggered from your configured deployment branch.
      parameters:
        - name: projectId
          in: path
          description: >-
            Your project ID. Can be copied from the [API
            keys](https://app.mintlify.com/settings/organization/api-keys) page
            in your dashboard.
          required: true
          schema:
            type: string
      responses:
        '202':
          description: A successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  statusId:
                    type: string
                    description: The status ID of the triggered updated.
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      description: >-
        The Authorization header expects a Bearer token. Use an admin API key
        (prefixed with `mint_`). This is a server-side secret key. Generate one
        on the [API keys
        page](https://app.mintlify.com/settings/organization/api-keys) in your
        dashboard.

````

## Related topics

- [Deployment permissions](/docs/dashboard/permissions.md)
- [GitHub](/docs/deploy/github.md)
- [OpenAPI setup](/docs/api-playground/openapi-setup.md)

Create docs.json
  Mintlify requires docs.json for site config/navigation.

  {
    "$schema": "https://mintlify.com/docs.json",
    "theme": "mint",
    "name": "London Coworking Event Finder",
    "colors": {
      "primary": "#0f766e"
    },
    "navigation": {
      "groups": [
        {
          "group": "Home",
          "pages": ["index"]
        }
      ]
    }
  }

  2. Create index.mdx

  ---
  title: "London Coworking Event Finder"
  description: "Find coworking spaces in London for events, meetings, and workshops."
  ---

  import { CoworkingFinder } from "/snippets/CoworkingFinder.jsx"

  <CoworkingFinder />

  3. Convert the app into snippets/CoworkingFinder.jsx
  Move the current app logic into a named React component. Mintlify supports custom React in MDX, but with
  constraints: named exports only, no external npm packages, no JSON imports, and component files must live
  in /snippets/.

  4. Preview Locally
  Install Mintlify CLI:

  npm i -g mint

  Then run:

  mint dev

  Mintlify previews at:

  http://localhost:3000

  5. Deploy
  Go to Mintlify onboarding, connect your GitHub repo, and install the Mintlify GitHub App. After that,
  Mintlify auto-deploys when you push changes. Your site will be available at:

  https://your-project-name.mintlify.app

  Useful official docs:

  - Mintlify Quickstart (https://www.mintlify.com/docs/quickstart)
  - docs.json settings (https://www.mintlify.com/docs/organize/settings)
  - React components in Mintlify (https://www.mintlify.com/docs/customize/react-components)
  - Deployments (https://www.mintlify.com/docs/deploy/deployments)
