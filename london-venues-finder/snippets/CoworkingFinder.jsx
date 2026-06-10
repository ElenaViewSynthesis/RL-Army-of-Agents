const londonCenter = {
  label: "the Maps search center",
  lat: 51.5047424,
  lng: -0.131072
};

const photoSets = {
  coworking: [
    ["https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=900&q=80", "Bright coworking lounge with desks and large windows"],
    ["https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=900&q=80", "Modern shared workspace with meeting areas"],
    ["https://images.unsplash.com/photo-1497366811353-6870744d04b2?auto=format&fit=crop&w=900&q=80", "Contemporary office breakout space with glass meeting rooms"],
    ["https://images.unsplash.com/photo-1604328698692-f76ea9498e76?auto=format&fit=crop&w=900&q=80", "Coworking desk area with warm lighting and plants"]
  ],
  meetings: [
    ["https://images.unsplash.com/photo-1517502884422-41eaead166d4?auto=format&fit=crop&w=900&q=80", "Meeting room prepared for a business session"],
    ["https://images.unsplash.com/photo-1542744173-8e7e53415bb0?auto=format&fit=crop&w=900&q=80", "Team gathered around a conference table"],
    ["https://images.unsplash.com/photo-1556761175-5973dc0f32e7?auto=format&fit=crop&w=900&q=80", "Business presentation in a polished meeting room"],
    ["https://images.unsplash.com/photo-1568992687947-868a62a9f521?auto=format&fit=crop&w=900&q=80", "Professional workshop session in a modern venue"]
  ],
  events: [
    ["https://images.unsplash.com/photo-1505373877841-8d25f7d46678?auto=format&fit=crop&w=900&q=80", "Conference audience watching a speaker"],
    ["https://images.unsplash.com/photo-1540575467063-178a50c2df87?auto=format&fit=crop&w=900&q=80", "Panel event with audience seating"],
    ["https://images.unsplash.com/photo-1511578314322-379afb476865?auto=format&fit=crop&w=900&q=80", "Modern event venue with seated attendees"],
    ["https://images.unsplash.com/photo-1515169067865-5387ec356754?auto=format&fit=crop&w=900&q=80", "Workshop setup with tables and presentation area"]
  ],
  lounge: [
    ["https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=900&q=80", "Plant-filled creative workspace lounge"],
    ["https://images.unsplash.com/photo-1497366858526-0766cadbe8fa?auto=format&fit=crop&w=900&q=80", "Green office lounge with casual seating"],
    ["https://images.unsplash.com/photo-1556761175-4b46a572b786?auto=format&fit=crop&w=900&q=80", "People collaborating in a casual coworking space"],
    ["https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=900&q=80", "Creative team meeting in a bright workspace"]
  ]
};

const makePhotos = (setName) => photoSets[setName].map(([src, alt]) => ({ src, alt }));

const spaces = [
  {
    id: "runway-east-bloomsbury",
    name: "Runway East Bloomsbury",
    area: "central",
    neighbourhood: "Bloomsbury",
    tube: "Holborn",
    lat: 51.518,
    lng: -0.1245,
    capacityLabel: "up to 120",
    fit: ["panel", "launch", "workshop"],
    photos: makePhotos("coworking"),
    summary: "Central workspace with meeting rooms, event support, and a documented 120-person event space.",
    website: "https://www.runwayea.st/event-spaces-location/bloomsbury",
    maps: "https://www.google.com/maps/search/?api=1&query=Runway+East+Bloomsbury+24-28+Bloomsbury+Way+London"
  },
  {
    id: "soho-works-180-strand",
    name: "Soho Works 180 Strand",
    area: "central",
    neighbourhood: "Strand",
    tube: "Temple",
    lat: 51.5126,
    lng: -0.1176,
    capacityLabel: "up to 200",
    fit: ["conference", "brand event", "panel"],
    photos: makePhotos("meetings"),
    summary: "Workspace and meeting-room venue with combinable spaces for larger gatherings, brand events, and conferences.",
    website: "https://api.production.sohohousedigital.com/sitecore/-/media/pdfs/soho-works/sw_meeting-room-brochure_2026.pdf",
    maps: "https://www.google.com/maps/search/?api=1&query=Soho+Works+180+Strand+London"
  },
  {
    id: "the-trampery-old-street",
    name: "The Trampery Old Street",
    area: "east",
    neighbourhood: "Old Street",
    tube: "Old Street",
    lat: 51.5255,
    lng: -0.0859,
    capacityLabel: "medium to large",
    fit: ["off-site", "conference", "workshop"],
    photos: makePhotos("events"),
    summary: "Creative workspace operator with an Old Street events hub, four meeting rooms, and 2000 sq ft of event space.",
    website: "https://thetrampery.com/spaces/",
    maps: "https://www.google.com/maps/search/?api=1&query=The+Trampery+Old+Street+London"
  },
  {
    id: "second-home-spitalfields",
    name: "Second Home Spitalfields",
    area: "east",
    neighbourhood: "Spitalfields",
    tube: "Liverpool Street",
    lat: 51.5205,
    lng: -0.071,
    capacityLabel: "small to midsize",
    fit: ["culture", "panel", "reception"],
    photos: makePhotos("lounge"),
    summary: "Design-led coworking and cultural venue with London venue hire and meeting-room options.",
    website: "https://secondhome.io/",
    maps: "https://www.google.com/maps/search/?api=1&query=Second+Home+Spitalfields+London"
  },
  {
    id: "work-life-holborn",
    name: "Work.Life Holborn",
    area: "central",
    neighbourhood: "Holborn",
    tube: "Holborn",
    lat: 51.5208,
    lng: -0.1167,
    capacityLabel: "small to midsize",
    fit: ["team social", "workshop", "meeting"],
    photos: makePhotos("coworking"),
    summary: "Flexible coworking with bookable meeting rooms and discounted event-space hire across London locations.",
    website: "https://work.life/coworking/london/holborn/",
    maps: "https://www.google.com/maps/search/?api=1&query=Work.Life+Holborn+20+Red+Lion+Street+London"
  },
  {
    id: "huckletree-soho",
    name: "Huckletree Soho",
    area: "central",
    neighbourhood: "Soho",
    tube: "Oxford Circus",
    lat: 51.5129,
    lng: -0.1346,
    capacityLabel: "small to midsize",
    fit: ["startup", "panel", "networking"],
    photos: makePhotos("meetings"),
    summary: "Central London workspace positioned for brands, scaling businesses, and investors to meet, work, and grow.",
    website: "https://www.huckletree.com/",
    maps: "https://www.google.com/maps/search/?api=1&query=Huckletree+Soho+London"
  },
  {
    id: "second-home-holland-park",
    name: "Second Home Holland Park",
    area: "west",
    neighbourhood: "Holland Park",
    tube: "Holland Park",
    lat: 51.5064,
    lng: -0.2043,
    capacityLabel: "small to midsize",
    fit: ["creative", "meeting", "reception"],
    photos: makePhotos("lounge"),
    summary: "West London creative coworking space with meeting rooms and venue-hire options.",
    website: "https://secondhome.io/",
    maps: "https://www.google.com/maps/search/?api=1&query=Second+Home+Holland+Park+London"
  },
  {
    id: "argyll-pall-mall",
    name: "Argyll - 78-79 Pall Mall",
    area: "central",
    neighbourhood: "St James's",
    tube: "Green Park",
    lat: 51.5057674,
    lng: -0.1360801,
    capacityLabel: "private offices 1-20 desks",
    fit: ["meeting", "office", "lounge"],
    photos: makePhotos("coworking"),
    summary: "Prestigious Pall Mall workspace in a Grade II listed townhouse with meeting rooms, business lounges, and Argyll coworking access across London.",
    website: "https://www.workargyll.com/properties/pall-mall/",
    maps: "https://www.google.com/maps/place/Argyll+-+78-79+Pall+Mall/@51.5057674,-0.1360801,17z"
  },
  {
    id: "argyll-north-audley-street",
    name: "Argyll - 20 North Audley Street",
    area: "central",
    neighbourhood: "Mayfair",
    tube: "Bond Street",
    lat: 51.5137,
    lng: -0.1524,
    capacityLabel: "up to 80",
    fit: ["coworking", "meeting", "event"],
    photos: makePhotos("meetings"),
    summary: "Mayfair coworking and business lounge location with meeting rooms, private call booths, and event spaces listed for up to 80 guests.",
    website: "https://www.workargyll.com/properties/north-audley-street/",
    maps: "https://www.google.com/maps/search/?api=1&query=Argyll+20+North+Audley+Street+London"
  },
  {
    id: "argyll-hill-street",
    name: "Argyll - 8-10 Hill Street",
    area: "central",
    neighbourhood: "Mayfair",
    tube: "Green Park",
    lat: 51.5089,
    lng: -0.1471,
    capacityLabel: "boutique workspace",
    fit: ["coworking", "meeting", "garden"],
    photos: makePhotos("lounge"),
    summary: "Boutique Mayfair townhouse workspace between Berkeley Square and the Royal Academy, with coworking access, meeting rooms, and a walled garden.",
    website: "https://www.workargyll.com/properties/hill-street",
    maps: "https://www.google.com/maps/search/?api=1&query=Argyll+8-10+Hill+Street+London"
  },
  {
    id: "argyll-84-brook-street",
    name: "Argyll - 84 Brook Street",
    area: "central",
    neighbourhood: "Mayfair",
    tube: "Bond Street",
    lat: 51.5127,
    lng: -0.1496,
    capacityLabel: "boutique townhouse",
    fit: ["office", "meeting", "lounge"],
    photos: makePhotos("meetings"),
    summary: "Bespoke Mayfair workspace with a boutique townhouse feel, suited to private offices, meetings, and refined lounge-based working.",
    website: "https://www.workargyll.com/properties/84-brook-street/",
    maps: "https://www.google.com/maps/search/?api=1&query=Argyll+84+Brook+Street+London"
  },
  {
    id: "clubhouse-st-james",
    name: "Clubhouse St James",
    area: "central",
    neighbourhood: "St James's",
    tube: "Piccadilly Circus",
    lat: 51.5078725,
    lng: -0.1356682,
    capacityLabel: "1-100",
    fit: ["coworking", "hot desk", "meeting"],
    photos: makePhotos("coworking"),
    summary: "Business members club and flexible workspace at 8 St James's Square, with coworking, hot desks, meeting rooms, and lounge space.",
    website: "https://www.theclubhouseoffices.com/locations",
    maps: "https://www.google.com/maps/place/Clubhouse+St+James/@51.5078725,-0.1356682,17z"
  },
  {
    id: "convene-sancroft-st-pauls",
    name: "Convene Sancroft, St. Paul's",
    area: "central",
    neighbourhood: "St Paul's",
    tube: "St Paul's",
    lat: 51.5150431,
    lng: -0.0996613,
    capacityLabel: "up to 1,200",
    fit: ["conference", "exhibition", "reception"],
    photos: makePhotos("events"),
    summary: "Large Paternoster Square meeting and event venue with seven meeting rooms, in-house catering, A/V support, and total capacity for up to 1,200 guests.",
    website: "https://convene.com/locations/london/sancroft-st-pauls/",
    maps: "https://www.google.com/maps/place/Convene+Sancroft,+St.+Paul's/@51.5150431,-0.0996613,17z"
  }
];

const sourceLinks = [
  ["Work.Life", "https://work.life/coworking/london/"],
  ["Huckletree", "https://www.huckletree.com/"],
  ["The Trampery", "https://thetrampery.com/spaces/"],
  ["Second Home", "https://secondhome.io/"],
  ["Runway East", "https://www.runwayea.st/event-spaces-location/bloomsbury"],
  ["Soho Works", "https://api.production.sohohousedigital.com/sitecore/-/media/pdfs/soho-works/sw_meeting-room-brochure_2026.pdf"],
  ["Argyll", "https://www.workargyll.com/properties/pall-mall/"],
  ["Clubhouse", "https://www.theclubhouseoffices.com/locations"],
  ["Convene", "https://convene.com/locations/london/sancroft-st-pauls/"]
];

function milesBetween(a, b) {
  const earthRadiusMiles = 3958.8;
  const toRadians = (degrees) => degrees * Math.PI / 180;
  const dLat = toRadians(b.lat - a.lat);
  const dLng = toRadians(b.lng - a.lng);
  const lat1 = toRadians(a.lat);
  const lat2 = toRadians(b.lat);
  const h = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;

  return 2 * earthRadiusMiles * Math.asin(Math.sqrt(h));
}

export function CoworkingFinder() {
  const [search, setSearch] = useState("");
  const [area, setArea] = useState("all");
  const [people, setPeople] = useState("");
  const [eventSize, setEventSize] = useState("Any");
  const [origin, setOrigin] = useState(londonCenter);
  const [usingUserLocation, setUsingUserLocation] = useState(false);
  const [locationNotice, setLocationNotice] = useState("");
  const [photoIndexes, setPhotoIndexes] = useState({});

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    return spaces
      .map((space) => ({
        ...space,
        distance: milesBetween(origin, space)
      }))
      .filter((space) => area === "all" || space.area === area)
      .filter((space) => {
        const haystack = [
          space.name,
          space.neighbourhood,
          space.tube,
          space.summary,
          ...space.fit
        ].join(" ").toLowerCase();

        return haystack.includes(query);
      })
      .sort((a, b) => a.distance - b.distance);
  }, [area, origin, search]);

  const useLocation = () => {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      setLocationNotice("Location is not available in this browser");
      return;
    }

    navigator.geolocation.getCurrentPosition((position) => {
      const candidate = {
        label: "your location",
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };

      if (milesBetween(candidate, londonCenter) > 80) {
        setOrigin(londonCenter);
        setUsingUserLocation(false);
        setLocationNotice("Outside London; showing central London distances");
        return;
      }

      setOrigin(candidate);
      setUsingUserLocation(true);
      setLocationNotice("");
    }, () => {
      setLocationNotice("Location permission was not granted");
    });
  };

  const setPhoto = (space, direction) => {
    const currentIndex = photoIndexes[space.id] || 0;
    const offset = direction === "next" ? 1 : -1;
    const nextIndex = (currentIndex + offset + space.photos.length) % space.photos.length;
    setPhotoIndexes((current) => ({ ...current, [space.id]: nextIndex }));
  };

  const updateEventSize = () => {
    const peopleCount = Number.parseInt(people, 10);
    setEventSize(Number.isNaN(peopleCount) ? "Any" : `${peopleCount} people`);
  };

  return (
    <main className="cwf">
      <section className="cwf-hero" aria-labelledby="cwf-title">
        <div>
          <p className="cwf-eyebrow">London event-ready coworking</p>
          <h1 id="cwf-title">Find a coworking space to host your next event.</h1>
        </div>
        <div className="cwf-hero-actions">
          <a className="cwf-secondary-link" href="https://www.google.com/maps/search/coworking+spaces/@51.5047424,-0.131072,14z?entry=ttu" target="_blank" rel="noreferrer">Open Maps results</a>
          <button className="cwf-secondary-button" type="button" onClick={useLocation}>Use my location</button>
        </div>
      </section>

      <section className="cwf-controls" aria-label="Filters">
        <label>
          <span>Search</span>
          <input type="search" value={search} placeholder="Space, area, panel, workshop" onChange={(event) => setSearch(event.target.value)} />
        </label>

        <label>
          <span>Area</span>
          <select value={area} onChange={(event) => setArea(event.target.value)}>
            <option value="all">All London</option>
            <option value="central">Central</option>
            <option value="east">East</option>
            <option value="west">West</option>
          </select>
        </label>

        <label>
          <span>People</span>
          <span className="cwf-capacity-control">
            <input type="number" min="1" max="250" value={people} placeholder="Any" onChange={(event) => setPeople(event.target.value)} onKeyDown={(event) => event.key === "Enter" && updateEventSize()} />
            <button type="button" onClick={updateEventSize}>Set</button>
          </span>
        </label>
      </section>

      <div className="cwf-status-row">
        <p>{filtered.length} {filtered.length === 1 ? "space" : "spaces"} {filtered.length === spaces.length ? "shown" : "match"}</p>
        <p>Event size: {eventSize}</p>
        <p>{locationNotice || (usingUserLocation ? "Sorted from your location" : `Sorted from ${origin.label}`)}</p>
      </div>

      <section className="cwf-space-list" aria-live="polite">
        {filtered.length === 0 ? (
          <div className="cwf-empty">No matching spaces. Clear filters to show all results.</div>
        ) : filtered.map((space) => {
          const photoIndex = Math.min(photoIndexes[space.id] || 0, space.photos.length - 1);
          const photo = space.photos[photoIndex];

          return (
            <article className="cwf-space-card" key={space.id}>
              <div>
                <h2>{space.name}</h2>
                <div className="cwf-photo-slider" aria-label={`Photos for ${space.name}`}>
                  <button className="cwf-slider-button" type="button" data-direction="previous" aria-label={`Previous photo for ${space.name}`} onClick={() => setPhoto(space, "previous")}>&lsaquo;</button>
                  <figure>
                    <img src={photo.src} alt={photo.alt} loading="lazy" />
                    <figcaption>{photoIndex + 1} / {space.photos.length}</figcaption>
                  </figure>
                  <button className="cwf-slider-button" type="button" data-direction="next" aria-label={`Next photo for ${space.name}`} onClick={() => setPhoto(space, "next")}>&rsaquo;</button>
                </div>
                <div className="cwf-meta">
                  <span className="cwf-pill">{space.neighbourhood}</span>
                  <span className="cwf-pill">{space.tube}</span>
                  <span className="cwf-pill">{space.distance.toFixed(1)} mi</span>
                  <span className="cwf-pill cwf-fit">{space.capacityLabel}</span>
                  {space.fit.map((tag) => <span className="cwf-pill" key={tag}>{tag}</span>)}
                </div>
                <p className="cwf-summary">{space.summary}</p>
              </div>
              <div className="cwf-actions" aria-label={`Actions for ${space.name}`}>
                <a className="cwf-primary" href={space.website} target="_blank" rel="noreferrer">Venue</a>
                <a href={space.maps} target="_blank" rel="noreferrer">Map</a>
              </div>
            </article>
          );
        })}
      </section>

      <footer className="cwf-footer">
        <span>Venue details change. Verify capacity, pricing, and availability with each operator.</span>
        {sourceLinks.map(([label, href]) => <a href={href} key={label} target="_blank" rel="noreferrer">{label}</a>)}
      </footer>

      <style>{`
        .cwf {
          --cw-bg: #f6f4ef;
          --cw-surface: #ffffff;
          --cw-ink: #1f2428;
          --cw-muted: #626b73;
          --cw-line: #d9d5ca;
          --cw-accent: #0f766e;
          --cw-accent-strong: #134e4a;
          --cw-warm: #b45309;
          width: min(1080px, 100%);
          margin: 0 auto;
          padding: 8px 0 32px;
          color: var(--cw-ink);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .cwf * { box-sizing: border-box; }
        .cwf a { color: inherit; }
        .cwf-hero {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 24px;
          padding: 28px 0 22px;
          border-bottom: 1px solid var(--cw-line);
        }
        .cwf-eyebrow {
          margin: 0 0 8px;
          color: var(--cw-accent-strong);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0;
          text-transform: uppercase;
        }
        .cwf h1 {
          max-width: 720px;
          margin: 0;
          font-size: clamp(2rem, 5vw, 4.4rem);
          line-height: 0.98;
          letter-spacing: 0;
        }
        .cwf button, .cwf input, .cwf select, .cwf-secondary-link { font: inherit; }
        .cwf-hero-actions {
          display: flex;
          flex: 0 0 auto;
          gap: 10px;
          align-items: center;
        }
        .cwf-secondary-button, .cwf-secondary-link {
          min-height: 44px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 8px;
          padding: 0 16px;
          font-weight: 800;
          cursor: pointer;
          white-space: nowrap;
        }
        .cwf-secondary-button {
          border: 1px solid var(--cw-accent);
          background: var(--cw-accent);
          color: white;
        }
        .cwf-secondary-link {
          border: 1px solid var(--cw-line);
          background: var(--cw-surface);
          color: var(--cw-ink);
          text-decoration: none;
        }
        .cwf-controls {
          display: grid;
          grid-template-columns: minmax(220px, 1fr) minmax(150px, 190px) minmax(120px, 150px);
          gap: 12px;
          padding: 20px 0 14px;
        }
        .cwf label {
          display: grid;
          gap: 7px;
          color: var(--cw-muted);
          font-size: 0.82rem;
          font-weight: 800;
        }
        .cwf input, .cwf select {
          width: 100%;
          min-height: 44px;
          border: 1px solid var(--cw-line);
          border-radius: 8px;
          background: var(--cw-surface);
          color: var(--cw-ink);
          padding: 0 12px;
          font-weight: 650;
        }
        .cwf-capacity-control {
          display: grid;
          grid-template-columns: minmax(72px, 1fr) auto;
          gap: 8px;
        }
        .cwf-capacity-control button {
          min-height: 44px;
          border: 1px solid var(--cw-ink);
          border-radius: 8px;
          background: var(--cw-ink);
          color: white;
          padding: 0 12px;
          font-weight: 850;
          cursor: pointer;
        }
        .cwf-status-row {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          color: var(--cw-muted);
          font-size: 0.9rem;
          font-weight: 750;
        }
        .cwf-status-row p { margin: 0 0 16px; }
        .cwf-space-list {
          display: grid;
          gap: 10px;
        }
        .cwf-space-card {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          gap: 18px;
          align-items: center;
          border: 1px solid var(--cw-line);
          border-radius: 8px;
          background: var(--cw-surface);
          padding: 18px;
        }
        .cwf-space-card h2 {
          margin: 0;
          font-size: 1.12rem;
          letter-spacing: 0;
        }
        .cwf-photo-slider {
          position: relative;
          margin-top: 12px;
          border-radius: 8px;
          overflow: hidden;
          border: 1px solid var(--cw-line);
          background: var(--cw-bg);
        }
        .cwf-photo-slider figure { margin: 0; }
        .cwf-photo-slider img {
          display: block;
          width: 100%;
          aspect-ratio: 16 / 7;
          object-fit: cover;
        }
        .cwf-photo-slider figcaption {
          position: absolute;
          right: 10px;
          bottom: 10px;
          border-radius: 999px;
          background: color-mix(in srgb, var(--cw-ink), transparent 20%);
          color: white;
          padding: 4px 8px;
          font-size: 0.76rem;
          font-weight: 850;
        }
        .cwf-slider-button {
          position: absolute;
          top: 50%;
          z-index: 1;
          width: 34px;
          height: 34px;
          border: 1px solid color-mix(in srgb, white, transparent 45%);
          border-radius: 999px;
          background: color-mix(in srgb, var(--cw-ink), transparent 22%);
          color: white;
          font-size: 1.45rem;
          font-weight: 900;
          line-height: 1;
          transform: translateY(-50%);
          cursor: pointer;
        }
        .cwf-slider-button[data-direction="previous"] { left: 10px; }
        .cwf-slider-button[data-direction="next"] { right: 10px; }
        .cwf-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin: 10px 0;
        }
        .cwf-pill {
          border: 1px solid var(--cw-line);
          border-radius: 999px;
          padding: 5px 9px;
          color: var(--cw-muted);
          font-size: 0.78rem;
          font-weight: 800;
        }
        .cwf-fit {
          border-color: color-mix(in srgb, var(--cw-warm), white 60%);
          color: var(--cw-warm);
        }
        .cwf-summary {
          margin: 0;
          color: var(--cw-muted);
          line-height: 1.45;
        }
        .cwf-actions {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 8px;
        }
        .cwf-actions a {
          min-height: 38px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border: 1px solid var(--cw-line);
          border-radius: 8px;
          padding: 0 12px;
          text-decoration: none;
          font-size: 0.86rem;
          font-weight: 850;
        }
        .cwf-actions a.cwf-primary {
          background: var(--cw-ink);
          border-color: var(--cw-ink);
          color: white;
        }
        .cwf-empty {
          border: 1px dashed var(--cw-line);
          border-radius: 8px;
          padding: 22px;
          color: var(--cw-muted);
          font-weight: 750;
        }
        .cwf-footer {
          display: flex;
          flex-wrap: wrap;
          gap: 10px 14px;
          margin-top: 22px;
          padding-top: 16px;
          border-top: 1px solid var(--cw-line);
          color: var(--cw-muted);
          font-size: 0.78rem;
          line-height: 1.4;
        }
        .cwf-footer span { flex-basis: 100%; }
        @media (max-width: 760px) {
          .cwf-hero, .cwf-status-row, .cwf-space-card { display: grid; }
          .cwf-hero { align-items: start; }
          .cwf-hero-actions, .cwf-secondary-button, .cwf-secondary-link, .cwf-actions, .cwf-actions a { width: 100%; }
          .cwf-hero-actions { display: grid; }
          .cwf-controls { grid-template-columns: 1fr; }
        }
      `}</style>
    </main>
  );
}
