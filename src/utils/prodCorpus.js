// prodCorpus.js — Sample corpus for the Production Lab: one long, frequently edited
// employee handbook plus a knowledge base of short IT / workplace articles.
// Everything is fictional. Kept as structured sections so both chunkers can use it.

export const HANDBOOK = {
  id: 'handbook',
  title: 'Employee Handbook',
  sections: [
    { id: '1', title: '1 Welcome', paras: [
      'Welcome to Northwind Labs. This handbook explains how we work, what we expect from each other, and where to find help when something is unclear.',
      'The handbook is owned by the People team and reviewed every quarter. Small corrections can be made at any time and take effect as soon as they are published.',
      'Everyone at Northwind may work from home up to 2 days per week once onboarding is complete. Team leads agree the exact days with their teams.',
      'If anything in this handbook conflicts with your employment contract, the contract wins. Ask the People team if you are unsure which applies.',
      'New starters get a buddy for their first 60 days. The buddy is your first stop for questions about tools, people and unwritten rules.',
    ] },
    { id: '2', title: '2 Working hours', paras: [
      'Core collaboration hours are 10:00 to 15:00 in your local time zone. Outside those hours you can arrange your day as you like.',
      'A standard working week is 37.5 hours. Overtime must be agreed in advance with your team lead and is recorded in the time tracker.',
      'Meetings should not be scheduled on Friday afternoons. Use that time for focused work, learning or tidying up the week.',
      'If you work across several time zones, publish your working hours on your calendar profile so colleagues know when to expect a reply.',
      'Breaks are not tracked. Take a proper lunch away from the screen and a short break at least every two hours.',
    ] },
    { id: '3', title: '3 Leave and time off', paras: [
      'Every full-time employee receives 15 days of paid annual leave per year, plus public holidays in their country of employment.',
      'Leave requests go through the HR portal at least two weeks in advance for absences longer than three days.',
      'Up to 5 unused leave days can be carried over into the next year. Carried-over days expire on 31 March.',
      'Sick leave does not count against annual leave. For absences longer than three consecutive days a doctor’s note is required.',
      'Parental leave, bereavement leave and volunteering days are described in the separate Family and Community policy.',
    ] },
    { id: '4', title: '4 Remote work', paras: [
      'Remote work is allowed up to 2 days per week. Fully remote arrangements need written approval from a director.',
      'When working remotely you must be reachable on chat during core hours and join video calls with your camera available.',
      'Working from another country for more than 10 working days per year requires approval from the People team because of tax rules.',
      'Home office equipment can be ordered once per year through the equipment catalogue. Chairs and monitors are delivered to your home address.',
      'Do not work from public places on confidential material unless you use a privacy screen and the company VPN.',
    ] },
    { id: '5', title: '5 Equipment', paras: [
      'Every employee receives a laptop on their first day. Laptops are refreshed every 3 years or earlier if they fail.',
      'You may install software from the self-service catalogue. Anything else needs a ticket to the IT service desk.',
      'Lost or stolen devices must be reported to the service desk within 24 hours so they can be locked remotely.',
      'Personal phones may be used for work email only after enrolling them in device management.',
      'Return all equipment on your last day. The service desk will send a prepaid return box if you work remotely.',
    ] },
    { id: '6', title: '6 Security', paras: [
      'Use the password manager for every work account. Passwords must be at least 14 characters and never reused.',
      'Multi-factor authentication is mandatory for email, the code repository and the HR portal.',
      'Report suspected phishing with the Report button in the mail client. Do not forward the suspicious message to colleagues.',
      'Customer data must stay in approved systems. Never paste customer data into personal tools or unapproved AI assistants.',
      'Security awareness training is completed during onboarding and refreshed every 12 months.',
    ] },
    { id: '7', title: '7 Travel', paras: [
      'Business travel is booked through the travel desk so that we can locate travellers in an emergency.',
      'Economy class is standard. Premium economy is allowed for flights longer than 6 hours.',
      'Hotels should be within walking distance or a short taxi ride of the office or venue you are visiting.',
      'Travel for conferences needs approval from your team lead at least four weeks before the event.',
      'Carry the travel desk emergency number with you. It is staffed 24 hours a day, every day of the year.',
    ] },
    { id: '8', title: '8 Benefits summary', paras: [
      'Health cover starts on your first day and includes dental and vision for you and your dependants.',
      'Annual leave is 15 days per year for full-time staff, pro-rated for part-time contracts.',
      'Each employee has a yearly learning budget for courses, books and conference tickets. See section 10 for the approval process.',
      'The wellbeing programme includes free counselling sessions and a gym membership discount.',
      'Benefits are reviewed every January. Changes are announced in the all-hands meeting and in this handbook.',
    ] },
    { id: '9', title: '9 Code of conduct', paras: [
      'Treat colleagues, customers and partners with respect. Harassment and discrimination are never acceptable.',
      'Raise concerns with your team lead, the People team or the anonymous ethics line. Retaliation against anyone who speaks up is prohibited.',
      'Declare any conflict of interest, including side projects that overlap with Northwind products.',
      'Gifts above a token value must be declared in the gift register within five working days.',
      'Speak for yourself on social media. Only the communications team speaks on behalf of Northwind.',
    ] },
    { id: '10', title: '10 Learning and growth', paras: [
      'Every employee can spend up to 5 working days per year on learning, in addition to the learning budget.',
      'Course and conference requests are approved by your team lead. Requests over the budget need director approval.',
      'Internal tech talks run every second Thursday. Anyone can propose a talk through the engineering wiki.',
      'Career conversations happen twice a year, separate from performance reviews.',
      'Mentoring is open to everyone. Sign up in the mentoring directory to be matched within a month.',
    ] },
    { id: '11', title: '11 Leaving Northwind', paras: [
      'The notice period is set in your contract. Tell your team lead first, then confirm in writing to the People team.',
      'Unused annual leave is paid out or taken during the notice period, as agreed with your team lead.',
      'Your accounts are disabled at 18:00 on your last working day. Move any personal files before then.',
      'An exit interview is offered to everyone. It is optional and the notes are kept confidential by the People team.',
      'Alumni are welcome at public tech talks and the yearly summer party.',
    ] },
    { id: '12', title: '12 Contacts', paras: [
      'IT service desk: open a ticket in the service portal or call extension 4357 during office hours.',
      'People team: people@northwind.example for questions about contracts, leave and benefits.',
      'Facilities: facilities@northwind.example for building access, desks and meeting rooms.',
      'Security team: security@northwind.example for incidents. For urgent incidents call the on-call number on the intranet home page.',
      'Handbook owner: the People team. Suggest an edit with the Edit button on any section of the intranet page.',
    ] },
  ],
}

// Knowledge base: 40 topics x 3 article kinds = 120 short documents.
const TOPICS = [
  'vpn', 'printer', 'laptop', 'monitor', 'wifi', 'email', 'calendar', 'badge', 'parking', 'desk booking',
  'password manager', 'video conferencing', 'shared drive', 'wiki', 'ticketing', 'headset', 'docking station',
  'mobile phone', 'software install', 'backup', 'antivirus', 'code repository', 'build server', 'test environment',
  'staging database', 'api gateway', 'log search', 'metrics dashboard', 'pager', 'feature flags',
  'container registry', 'secrets vault', 'single sign-on', 'travel booking', 'meeting rooms', 'mail room',
  'cafeteria', 'gym access', 'bike storage', 'first aid',
]
const KINDS = [
  { key: 'setup', label: 'Setup guide', paras: t => [
    `How to set up ${t}. Open the self-service portal, search the catalogue for ${t} and press Request. Most ${t} requests are approved automatically.`,
    `After approval the ${t} setup steps arrive by email. Follow them in order and restart your laptop if the ${t} setup asks for it.`,
    `If your team needs ${t} for a new project, the team lead can request ${t} access for the whole group at once.`,
    `The ${t} owner is listed on the service catalogue page together with the support hours for ${t}.`,
  ] },
  { key: 'trouble', label: 'Troubleshooting', paras: t => [
    `Troubleshooting ${t} problems. Most ${t} issues are fixed by signing out, clearing the local cache and signing in again.`,
    `If ${t} still fails, check the status page. Known ${t} outages are posted there with an expected recovery time.`,
    `Collect the error message and a screenshot before opening a ${t} ticket. It halves the time to resolve ${t} incidents.`,
    `Escalate to the ${t} on-call engineer only when the ${t} outage blocks a customer-facing release.`,
  ] },
  { key: 'faq', label: 'FAQ', paras: t => [
    `Frequently asked questions about ${t}. Who can use ${t}? Every employee after onboarding, unless their role excludes ${t}.`,
    `Is ${t} available outside office hours? Yes, ${t} is available around the clock, but support for ${t} follows office hours.`,
    `Where is the ${t} policy? The ${t} policy lives on the intranet next to the ${t} service page.`,
    `Can contractors use ${t}? Contractors can use ${t} when their sponsor approves it in the access portal.`,
  ] },
]

export const KB = []
TOPICS.forEach((t, ti) => KINDS.forEach(k => {
  KB.push({ id: `kb-${ti}-${k.key}`, title: `${k.label}: ${t}`, topic: t, kind: k.key,
    sections: [{ id: '1', title: `${k.label}: ${t}`, paras: k.paras(t) }] })
}))

// Retrieval questions with their expected document (and section for the handbook).
export const QUESTIONS = [
  ...KB.map(d => ({ q: `${{ setup: 'how to set up', trouble: 'troubleshooting problems with', faq: 'frequently asked questions about' }[d.kind]} ${d.topic}`, docId: d.id })),
  { q: 'how many days of paid annual leave do I get', docId: 'handbook', section: '3' },
  { q: 'how many days per week can I work from home remotely', docId: 'handbook', section: '4' },
  { q: 'how often are laptops refreshed', docId: 'handbook', section: '5' },
  { q: 'minimum password length characters', docId: 'handbook', section: '6' },
  { q: 'premium economy flights longer than', docId: 'handbook', section: '7' },
  { q: 'gifts declared in the gift register', docId: 'handbook', section: '9' },
  { q: 'internal tech talks thursday', docId: 'handbook', section: '10' },
  { q: 'accounts disabled on last working day', docId: 'handbook', section: '11' },
]

export const cloneDoc = d => ({ ...d, sections: d.sections.map(s => ({ ...s, paras: [...s.paras] })) })
