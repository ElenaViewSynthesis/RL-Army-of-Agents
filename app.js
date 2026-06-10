const londonCenter = {
  label: "the Maps search center",
  lat: 51.5047424,
  lng: -0.131072
};

const spaces = [
  {
    id: "runway-east-bloomsbury",
    name: "Runway East Bloomsbury",
    area: "central",
    neighbourhood: "Bloomsbury",
    tube: "Holborn",
    lat: 51.518,
    lng: -0.1245,
    capacity: 120,
    capacityLabel: "up to 120",
    fit: ["panel", "launch", "workshop"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=900&q=80",
        alt: "Bright coworking lounge with desks and large windows"
      },
      {
        src: "https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=900&q=80",
        alt: "Modern shared workspace with meeting areas"
      },
      {
        src: "https://images.unsplash.com/photo-1497366811353-6870744d04b2?auto=format&fit=crop&w=900&q=80",
        alt: "Contemporary office breakout space with glass meeting rooms"
      },
      {
        src: "https://images.unsplash.com/photo-1604328698692-f76ea9498e76?auto=format&fit=crop&w=900&q=80",
        alt: "Coworking desk area with warm lighting and plants"
      }
    ],
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
    capacity: 200,
    capacityLabel: "up to 200",
    fit: ["conference", "brand event", "panel"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1517502884422-41eaead166d4?auto=format&fit=crop&w=900&q=80",
        alt: "Large meeting room prepared for a business session"
      },
      {
        src: "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?auto=format&fit=crop&w=900&q=80",
        alt: "Team gathered around a conference table"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-5973dc0f32e7?auto=format&fit=crop&w=900&q=80",
        alt: "Business presentation in a polished meeting room"
      },
      {
        src: "https://images.unsplash.com/photo-1568992687947-868a62a9f521?auto=format&fit=crop&w=900&q=80",
        alt: "Professional workshop session in a modern venue"
      }
    ],
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
    capacity: 140,
    capacityLabel: "medium to large",
    fit: ["off-site", "conference", "workshop"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1511578314322-379afb476865?auto=format&fit=crop&w=900&q=80",
        alt: "Event audience seated in a modern venue"
      },
      {
        src: "https://images.unsplash.com/photo-1515169067865-5387ec356754?auto=format&fit=crop&w=900&q=80",
        alt: "Workshop setup with tables and presentation space"
      },
      {
        src: "https://images.unsplash.com/photo-1505373877841-8d25f7d46678?auto=format&fit=crop&w=900&q=80",
        alt: "Conference audience watching a speaker"
      },
      {
        src: "https://images.unsplash.com/photo-1540575467063-178a50c2df87?auto=format&fit=crop&w=900&q=80",
        alt: "Panel event with audience seating"
      }
    ],
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
    capacity: 100,
    capacityLabel: "small to midsize",
    fit: ["culture", "panel", "reception"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=900&q=80",
        alt: "Plant-filled creative workspace lounge"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-4b46a572b786?auto=format&fit=crop&w=900&q=80",
        alt: "People collaborating in a casual coworking space"
      },
      {
        src: "https://images.unsplash.com/photo-1497366858526-0766cadbe8fa?auto=format&fit=crop&w=900&q=80",
        alt: "Green office lounge with casual seating"
      },
      {
        src: "https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=900&q=80",
        alt: "Creative team meeting in a bright workspace"
      }
    ],
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
    capacity: 80,
    capacityLabel: "small to midsize",
    fit: ["team social", "workshop", "meeting"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=900&q=80",
        alt: "Open coworking area with desks and natural light"
      },
      {
        src: "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=900&q=80",
        alt: "Small group workshop around a table"
      },
      {
        src: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=900&q=80",
        alt: "Team working together in a flexible office"
      },
      {
        src: "https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=900&q=80",
        alt: "Workshop group reviewing notes in a meeting"
      }
    ],
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
    capacity: 70,
    capacityLabel: "small to midsize",
    fit: ["startup", "panel", "networking"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=900&q=80",
        alt: "Informal business meeting in a shared workspace"
      },
      {
        src: "https://images.unsplash.com/photo-1551836022-4c4c79ecde51?auto=format&fit=crop&w=900&q=80",
        alt: "Startup team discussing plans in an office"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-129418cb2dfe?auto=format&fit=crop&w=900&q=80",
        alt: "Networking discussion in a modern work lounge"
      }
    ],
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
    capacity: 90,
    capacityLabel: "small to midsize",
    fit: ["creative", "meeting", "reception"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=900&q=80",
        alt: "Creative team meeting in a bright office"
      },
      {
        src: "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=900&q=80",
        alt: "People attending a collaborative business event"
      },
      {
        src: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=900&q=80",
        alt: "Creative team collaborating with laptops"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-4b46a572b786?auto=format&fit=crop&w=900&q=80",
        alt: "Casual coworking area with people collaborating"
      }
    ],
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
    capacity: 40,
    capacityLabel: "private offices 1-20 desks",
    fit: ["meeting", "office", "lounge"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=900&q=80",
        alt: "Elegant shared office with open seating"
      },
      {
        src: "https://images.unsplash.com/photo-1497366412874-3415097a27e7?auto=format&fit=crop&w=900&q=80",
        alt: "Traditional meeting room with natural light"
      },
      {
        src: "https://images.unsplash.com/photo-1497366811353-6870744d04b2?auto=format&fit=crop&w=900&q=80",
        alt: "Modern office lounge with glass meeting rooms"
      },
      {
        src: "https://images.unsplash.com/photo-1604328698692-f76ea9498e76?auto=format&fit=crop&w=900&q=80",
        alt: "Warm coworking area with desks and plants"
      }
    ],
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
    capacity: 80,
    capacityLabel: "up to 80",
    fit: ["coworking", "meeting", "event"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=900&q=80",
        alt: "Open coworking space with desks and daylight"
      },
      {
        src: "https://images.unsplash.com/photo-1517502884422-41eaead166d4?auto=format&fit=crop&w=900&q=80",
        alt: "Meeting room arranged for a business session"
      },
      {
        src: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=900&q=80",
        alt: "Team working together in a flexible office"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-5973dc0f32e7?auto=format&fit=crop&w=900&q=80",
        alt: "Presentation setup in a modern meeting room"
      }
    ],
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
    capacity: 50,
    capacityLabel: "boutique workspace",
    fit: ["coworking", "meeting", "garden"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=900&q=80",
        alt: "Plant-filled flexible workspace lounge"
      },
      {
        src: "https://images.unsplash.com/photo-1497366858526-0766cadbe8fa?auto=format&fit=crop&w=900&q=80",
        alt: "Green office lounge with casual seating"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=900&q=80",
        alt: "Informal business meeting in a shared workspace"
      },
      {
        src: "https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=900&q=80",
        alt: "Workshop group reviewing notes in a meeting"
      }
    ],
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
    capacity: 45,
    capacityLabel: "boutique townhouse",
    fit: ["office", "meeting", "lounge"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=900&q=80",
        alt: "Team working in a modern office"
      },
      {
        src: "https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=900&q=80",
        alt: "Bright office meeting with a creative team"
      },
      {
        src: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=900&q=80",
        alt: "Collaborative team working with laptops"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-129418cb2dfe?auto=format&fit=crop&w=900&q=80",
        alt: "Networking conversation in a modern work lounge"
      }
    ],
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
    capacity: 100,
    capacityLabel: "1-100",
    fit: ["coworking", "hot desk", "meeting"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=900&q=80",
        alt: "Bright flexible workspace with desks and windows"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-4b46a572b786?auto=format&fit=crop&w=900&q=80",
        alt: "People collaborating in a coworking lounge"
      },
      {
        src: "https://images.unsplash.com/photo-1517502884422-41eaead166d4?auto=format&fit=crop&w=900&q=80",
        alt: "Meeting room prepared for a business session"
      },
      {
        src: "https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=900&q=80",
        alt: "Informal meeting in a shared workspace"
      }
    ],
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
    capacity: 1200,
    capacityLabel: "up to 1,200",
    fit: ["conference", "exhibition", "reception"],
    photos: [
      {
        src: "https://images.unsplash.com/photo-1505373877841-8d25f7d46678?auto=format&fit=crop&w=900&q=80",
        alt: "Large conference audience watching a speaker"
      },
      {
        src: "https://images.unsplash.com/photo-1540575467063-178a50c2df87?auto=format&fit=crop&w=900&q=80",
        alt: "Panel event with audience seating"
      },
      {
        src: "https://images.unsplash.com/photo-1511578314322-379afb476865?auto=format&fit=crop&w=900&q=80",
        alt: "Modern event venue with seated attendees"
      },
      {
        src: "https://images.unsplash.com/photo-1515169067865-5387ec356754?auto=format&fit=crop&w=900&q=80",
        alt: "Workshop setup with tables and presentation area"
      }
    ],
    summary: "Large Paternoster Square meeting and event venue with seven meeting rooms, in-house catering, A/V support, and total capacity for up to 1,200 guests.",
    website: "https://convene.com/locations/london/sancroft-st-pauls/",
    maps: "https://www.google.com/maps/place/Convene+Sancroft,+St.+Paul's/@51.5150431,-0.0996613,17z"
  }
];

let origin = londonCenter;
let usingUserLocation = false;
let locationNotice = "";
const photoIndexes = new Map();

const list = document.querySelector("#spaceList");
const resultCount = document.querySelector("#resultCount");
const distanceMode = document.querySelector("#distanceMode");
const searchInput = document.querySelector("#searchInput");
const areaSelect = document.querySelector("#areaSelect");
const peopleInput = document.querySelector("#peopleInput");
const peopleButton = document.querySelector("#peopleButton");
const capacityNote = document.querySelector("#capacityNote");
const locationButton = document.querySelector("#locationButton");

const formatDistance = (distance) => `${distance.toFixed(1)} mi`;

function getPhotoIndex(space) {
  const savedIndex = photoIndexes.get(space.id) || 0;
  return Math.min(savedIndex, space.photos.length - 1);
}

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

function getFilteredSpaces() {
  const query = searchInput.value.trim().toLowerCase();
  const area = areaSelect.value;

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
}

function render() {
  const filtered = getFilteredSpaces();
  const label = filtered.length === 1 ? "space" : "spaces";
  const suffix = filtered.length === spaces.length ? "shown" : "match";
  resultCount.textContent = `${filtered.length} ${label} ${suffix}`;
  distanceMode.textContent = locationNotice || (usingUserLocation
    ? `Sorted from your location`
    : `Sorted from ${origin.label}`);

  if (!filtered.length) {
    list.innerHTML = `<div class="empty">No matching spaces. Clear filters to show all results.</div>`;
    return;
  }

  list.innerHTML = filtered.map((space) => `
    <article class="space-card">
      <div>
        <h2>${space.name}</h2>
        <div class="photo-slider" aria-label="Photos for ${space.name}">
          <button class="slider-button" type="button" data-space-id="${space.id}" data-direction="previous" aria-label="Previous photo for ${space.name}">&lsaquo;</button>
          <figure>
            <img src="${space.photos[getPhotoIndex(space)].src}" alt="${space.photos[getPhotoIndex(space)].alt}" loading="lazy">
            <figcaption>${getPhotoIndex(space) + 1} / ${space.photos.length}</figcaption>
          </figure>
          <button class="slider-button" type="button" data-space-id="${space.id}" data-direction="next" aria-label="Next photo for ${space.name}">&rsaquo;</button>
        </div>
        <div class="meta">
          <span class="pill">${space.neighbourhood}</span>
          <span class="pill">${space.tube}</span>
          <span class="pill">${formatDistance(space.distance)}</span>
          <span class="pill fit">${space.capacityLabel}</span>
          ${space.fit.map((tag) => `<span class="pill">${tag}</span>`).join("")}
        </div>
        <p class="summary">${space.summary}</p>
      </div>
      <div class="actions" aria-label="Actions for ${space.name}">
        <a class="primary" href="${space.website}" target="_blank" rel="noreferrer">Venue</a>
        <a href="${space.maps}" target="_blank" rel="noreferrer">Map</a>
      </div>
    </article>
  `).join("");
}

function useLocation() {
  if (!navigator.geolocation) {
    distanceMode.textContent = "Location is not available in this browser";
    return;
  }

  locationButton.disabled = true;
  locationButton.textContent = "Locating...";

  navigator.geolocation.getCurrentPosition((position) => {
    const candidate = {
      label: "your location",
      lat: position.coords.latitude,
      lng: position.coords.longitude
    };
    const distanceFromLondon = milesBetween(candidate, londonCenter);

    if (distanceFromLondon > 80) {
      origin = londonCenter;
      usingUserLocation = false;
      locationNotice = "Outside London; showing central London distances";
    } else {
      origin = candidate;
      usingUserLocation = true;
      locationNotice = "";
    }

    locationButton.disabled = false;
    locationButton.textContent = "Use my location";
    render();
  }, () => {
    locationButton.disabled = false;
    locationButton.textContent = "Use my location";
    locationNotice = "Location permission was not granted";
    render();
  });
}

[searchInput, areaSelect].forEach((control) => {
  control.addEventListener("input", () => {
    locationNotice = "";
    render();
  });
});

locationButton.addEventListener("click", useLocation);

peopleButton.addEventListener("click", () => {
  const peopleCount = Number.parseInt(peopleInput.value, 10);
  capacityNote.textContent = Number.isNaN(peopleCount)
    ? "Event size: Any"
    : `Event size: ${peopleCount} people`;
});

peopleInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    peopleButton.click();
  }
});

list.addEventListener("click", (event) => {
  const button = event.target.closest("[data-space-id][data-direction]");
  if (!button) {
    return;
  }

  const space = spaces.find((item) => item.id === button.dataset.spaceId);
  if (!space) {
    return;
  }

  const currentIndex = getPhotoIndex(space);
  const offset = button.dataset.direction === "next" ? 1 : -1;
  const nextIndex = (currentIndex + offset + space.photos.length) % space.photos.length;
  photoIndexes.set(space.id, nextIndex);
  render();
});

render();
