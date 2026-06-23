# London Venues Finder

London Venues Finder helps organisers find coworking spaces and flexible venues near London for meetups, events, workshops, and hackathons. Users can enter an estimated number of attendees, compare suitable spaces by capacity and location, and open a map view to see what is nearby.

## Purpose

This project is designed to make venue discovery faster and easier for people planning community, startup, student, and tech events. Instead of checking lots of venue pages manually, organisers can start with attendee numbers and event type, then shortlist spaces that match their needs.

## Core Features

- Search coworking spaces and flexible venues around London.
- Filter venues by estimated attendee capacity.
- Compare spaces for meetups, events, workshops, panels, and hackathons.
- Open a map view to check nearby transport, cafes, restaurants, hotels, and useful amenities.
- Store venue details such as neighbourhood, closest station, capacity, website, and event fit.

## Run With Mintlify

This folder includes the Mintlify documentation setup for the London Coworking Event Finder:

- `docs.json` controls the Mintlify site configuration and navigation.
- `index.mdx` is the Mintlify landing page.
- `snippets/CoworkingFinder.jsx` contains the interactive React component used by the page.

To preview the Mintlify site locally, install the Mintlify CLI:

```bash
npm i -g mint
```

Then run the local dev server from this folder:

```bash
cd london-venues-finder
mint dev
```

Mintlify will start a local preview at:

```text
http://localhost:3000
```

To deploy, connect the GitHub repository to Mintlify, install the Mintlify GitHub App, and make sure this folder is included in the deployment branch. After setup, Mintlify will redeploy when changes are pushed to the configured branch.

## Venue Data

| Field | Description |
| --- | --- |
| Venue name | Name of the coworking space or event venue |
| Area | London area, such as central, east, west, north, or south |
| Neighbourhood | Local area, such as Soho, Shoreditch, Holborn, or King's Cross |
| Closest station | Nearby Tube, rail, or Overground station |
| Estimated capacity | Approximate number of attendees the venue can host |
| Event fit | Meetup, hackathon, panel, workshop, launch, or networking |
| Website | Link to the venue or booking page |
| Map link | Link that opens the location in Google Maps |

## Example Capacity Guide

| Event type | Estimated capacity range |
| --- | --- |
| Small meetup | 20-50 attendees |
| Workshop | 30-80 attendees |
| Networking event | 50-120 attendees |
| Hackathon | 50-200 attendees |
| Larger event | 150+ attendees |

Capacity should be treated as an estimate until confirmed with the venue, because room layout, catering, equipment, and seating plans can change the real number of attendees a space can support.
