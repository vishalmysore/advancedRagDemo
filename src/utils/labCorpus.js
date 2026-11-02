// labCorpus.js — Built-in sample corpus + question/evidence pairs for the Chunking Lab.
// The handbook is fictional. It is deliberately written like real policy documents:
// numbered provisions, definitions that other provisions depend on, and explicit
// cross-references ("see Section 1.3"), because that is exactly where naive chunking breaks.

export const SAMPLE_DOC = `# Halcyon Ridge Observatory: Visitor and Operations Handbook

## 1 Definitions

### 1.1 Dark window
A dark window is any period of at least 90 minutes between the end of astronomical twilight and moonrise. The duty astronomer publishes tonight's dark window on the board in the Visitor Hut by 16:00. All timing rules in this handbook are measured against the published dark window, not against sunset.

### 1.2 Dome crew
The dome crew consists of one telescope operator and one safety lead. A dome crew is assigned to a single tour group for the whole evening and may not be split between groups. Volunteers can assist the crew but never count toward it.

### 1.3 Red-light zone
The red-light zone covers the upper terrace, the dome floor and the north stairway. Only red lamps of 5 lumens or less are permitted inside the red-light zone. White light of any kind, including phone screens at full brightness, must be switched off before entering it.

### 1.4 Guest
Guest means any visitor who is not a member of staff or a registered researcher. Registered researchers follow the separate research protocol and are not counted in tour group sizes.

## 2 Access and Arrival

### 2.1 Gate hours
The lower gate opens at 17:00. It closes to arriving vehicles 30 minutes before the start of the dark window defined in Section 1.1. Departing vehicles may leave at any time with headlights on low beam.

### 2.2 Parking
Vehicles must park in Lot B with headlights off after the gate closes. Lot A is reserved for the dome crew and for the observatory shuttle. Drivers who arrive with a trailer are directed to the gravel apron beside Lot B.

### 2.3 Check-in
Every guest checks in at the Visitor Hut and receives a red lamp that meets the limit in Section 1.3. Check-in closes 10 minutes before the tour starts. Guests who lose their lamp can collect a replacement from the safety lead.

### 2.4 Late arrivals
Guests arriving after the gate closes are escorted on foot from Lot B by the safety lead. Late guests may not join a tour that is already in progress and are offered a seat in the planetarium room instead.

## 3 Night Tours

### 3.1 Group size
A night tour carries at most 14 guests per dome crew. Groups larger than 14 are split before check-in, and each part receives its own dome crew as described in Section 1.2. Infants carried in a sling count toward the limit.

### 3.2 Duration
A standard tour lasts 75 minutes. Extended tours of 120 minutes run only during a dark window. The telescope operator may shorten any tour by up to 15 minutes if the target object sets early.

### 3.3 Route
Tours start at the Visitor Hut, cross the upper terrace and finish on the dome floor. The route is fully inside the red-light zone after the terrace gate. Guests who need step-free access use the service ramp on the east side.

### 3.4 Children
Children under 10 must hold an adult's hand inside the red-light zone (see Section 1.3). Children under 6 may join standard tours but not extended tours.

### 3.5 Cancellations
A tour is cancelled if cloud cover exceeds 60 percent at the 18:30 forecast check described in Section 6.1. Cancellations are announced by text message at least 90 minutes before the tour.

## 4 Equipment and Telescope Use

### 4.1 Main telescope
The 1.2-metre reflector is operated only by the telescope operator. Guests must never touch the mount, the focuser or the dome controls.

### 4.2 Eyepiece rotation
Guests view through the eyepiece in turns of 40 seconds. On extended tours each guest receives at least six turns.

### 4.3 Personal cameras
Phone cameras may be used on the terrace but never at the eyepiece. Flash photography is banned everywhere in the red-light zone. Tripods are allowed on the terrace only.

### 4.4 Cold limit
The dome is closed when the outside temperature falls below minus 18 degrees Celsius, because the mirror drive stalls. When the cold limit is reached the tour continues in the planetarium room.

## 5 Safety

### 5.1 Footwear
Closed shoes with grip soles are required on the north stairway. Sandals and heels are not permitted beyond the Visitor Hut after dark.

### 5.2 Emergency kit
Each dome crew carries a thermal blanket, a first-aid pouch and a handheld radio tuned to channel 7. The safety lead checks the kit at check-in and records the check on the tour sheet.

### 5.3 Evacuation
In an evacuation, guests leave by the north stairway and gather at the Lot B marker post. The safety lead counts guests against the tour sheet before anyone drives away.

### 5.4 Lightning
Any lightning within 15 kilometres ends all outdoor activity for 30 minutes after the last strike. During a lightning hold, guests wait in the Visitor Hut.

## 6 Weather and Scheduling

### 6.1 Forecast checks
The safety lead reads the cloud forecast at 18:30 and again at 21:00. The 21:00 check only affects extended tours.

### 6.2 Wind
Above 45 km/h of wind the dome shutter stays closed and tours move to the planetarium room. Wind is measured at the mast on the north stairway.

### 6.3 Moon nights
On nights with more than 80 percent lunar illumination, tours focus on the Moon and the planets. Deep-sky targets are not scheduled on moon nights.

### 6.4 Make-up tours
Guests whose tour was cancelled under Section 3.5 may rebook any tour within 60 days. A make-up booking keeps the original group size.
`

// Question → evidence pairs. Every evidence string is an exact substring of SAMPLE_DOC.
// "train" pairs play the role of the LLM-generated question/evidence supervision that
// QESChunker optimises on. "test" pairs are held out: QES never sees them.
export const SAMPLE_QE = [
  // ── train ──────────────────────────────────────────────────────────────
  { split: 'train', q: 'What counts as a dark window?',
    evidence: ['A dark window is any period of at least 90 minutes between the end of astronomical twilight and moonrise.'] },
  { split: 'train', q: 'Who is on the dome crew?',
    evidence: ['The dome crew consists of one telescope operator and one safety lead.'] },
  { split: 'train', q: 'When does the lower gate close to arriving cars?',
    evidence: ['It closes to arriving vehicles 30 minutes before the start of the dark window defined in Section 1.1.',
               'A dark window is any period of at least 90 minutes between the end of astronomical twilight and moonrise.'] },
  { split: 'train', q: 'How long is a standard tour?',
    evidence: ['A standard tour lasts 75 minutes.'] },
  { split: 'train', q: 'What happens to tours when it is very windy?',
    evidence: ['Above 45 km/h of wind the dome shutter stays closed and tours move to the planetarium room.'] },
  { split: 'train', q: 'Can I use my phone camera at the eyepiece?',
    evidence: ['Phone cameras may be used on the terrace but never at the eyepiece.'] },
  { split: 'train', q: 'What is in the emergency kit?',
    evidence: ['Each dome crew carries a thermal blanket, a first-aid pouch and a handheld radio tuned to channel 7.'] },
  { split: 'train', q: 'When is a tour cancelled for clouds, and who checks?',
    evidence: ['A tour is cancelled if cloud cover exceeds 60 percent at the 18:30 forecast check described in Section 6.1.',
               'The safety lead reads the cloud forecast at 18:30 and again at 21:00.'] },
  { split: 'train', q: 'How long does a lightning hold last?',
    evidence: ['Any lightning within 15 kilometres ends all outdoor activity for 30 minutes after the last strike.'] },
  { split: 'train', q: 'Where do vehicles park after the gate closes?',
    evidence: ['Vehicles must park in Lot B with headlights off after the gate closes.'] },
  { split: 'train', q: 'How long is each turn at the eyepiece?',
    evidence: ['Guests view through the eyepiece in turns of 40 seconds.'] },
  { split: 'train', q: 'Why does the dome close in extreme cold?',
    evidence: ['The dome is closed when the outside temperature falls below minus 18 degrees Celsius, because the mirror drive stalls.'] },

  // ── test (held out) ────────────────────────────────────────────────────
  { split: 'test', q: 'What lamp brightness does the check-in desk hand out?',
    evidence: ['Every guest checks in at the Visitor Hut and receives a red lamp that meets the limit in Section 1.3.',
               'Only red lamps of 5 lumens or less are permitted inside the red-light zone.'] },
  { split: 'test', q: 'A group of 20 friends arrives. How many staff will be assigned to them?',
    evidence: ['Groups larger than 14 are split before check-in, and each part receives its own dome crew as described in Section 1.2.',
               'The dome crew consists of one telescope operator and one safety lead.'] },
  { split: 'test', q: 'My 8-year-old is coming. What rule applies on the terrace?',
    evidence: ['Children under 10 must hold an adult\'s hand inside the red-light zone (see Section 1.3).',
               'The red-light zone covers the upper terrace, the dome floor and the north stairway.'] },
  { split: 'test', q: 'Can a 5-year-old join a 120-minute tour?',
    evidence: ['Extended tours of 120 minutes run only during a dark window.',
               'Children under 6 may join standard tours but not extended tours.'] },
  { split: 'test', q: 'If my tour is cancelled because of clouds, can I rebook, and how cloudy is too cloudy?',
    evidence: ['Guests whose tour was cancelled under Section 3.5 may rebook any tour within 60 days.',
               'A tour is cancelled if cloud cover exceeds 60 percent at the 18:30 forecast check described in Section 6.1.'] },
  { split: 'test', q: 'What shoes do I need for the evacuation route?',
    evidence: ['In an evacuation, guests leave by the north stairway and gather at the Lot B marker post.',
               'Closed shoes with grip soles are required on the north stairway.'] },
  { split: 'test', q: 'I arrive after the gate closes. Where do I park and who meets me?',
    evidence: ['Vehicles must park in Lot B with headlights off after the gate closes.',
               'Guests arriving after the gate closes are escorted on foot from Lot B by the safety lead.'] },
  { split: 'test', q: 'Is flash photography allowed on the dome floor?',
    evidence: ['Flash photography is banned everywhere in the red-light zone.',
               'The red-light zone covers the upper terrace, the dome floor and the north stairway.'] },
  { split: 'test', q: 'How many eyepiece turns do guests get on an extended tour, and how long is each?',
    evidence: ['On extended tours each guest receives at least six turns.',
               'Guests view through the eyepiece in turns of 40 seconds.'] },
  { split: 'test', q: 'Where do tours go if it is too cold or too windy?',
    evidence: ['When the cold limit is reached the tour continues in the planetarium room.',
               'Above 45 km/h of wind the dome shutter stays closed and tours move to the planetarium room.'] },
  { split: 'test', q: 'Does the 21:00 forecast check affect a standard tour?',
    evidence: ['The 21:00 check only affects extended tours.'] },
  { split: 'test', q: 'Who can operate the telescope?',
    evidence: ['The 1.2-metre reflector is operated only by the telescope operator.'] },
  { split: 'test', q: 'What white light is allowed on the north stairway?',
    evidence: ['White light of any kind, including phone screens at full brightness, must be switched off before entering it.',
               'The red-light zone covers the upper terrace, the dome floor and the north stairway.'] },
]
