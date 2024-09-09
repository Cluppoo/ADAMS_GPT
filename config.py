operator_config = {'is equal to' : 'ends',
                   'is not equal to' : 'not',
                   'contains' : 'contains',
                   'includes' : 'infolder'}


document_types = [
    "2.206 Acknowledgment Letter",
    "2.206 Director Decision",
    "2.206 Petition",
    "Acceptance Review Letter",
    "ACQ-Amendment of Solicitation/Modification of Contract, SF Form 30",
    "ACQ-Contract",
    "ACQ-Contract Closeout (Final Notice Letter)",
    "ACQ-Contract Deliverable",
    "ACQ-Contract Execution Correspondence",
    "ACQ-Contract Final Report",
    "ACQ-Contract Modification",
    "ACQ-Contract Modification Execution Correspondence",
    "ACQ-Contract Monthly Status Report",
    "ACQ-Contract Solicitation",
    "ACQ-Contract Status Report",
    "ACQ-Contract Task Order",
    "ACQ-Contract Task Order Modification",
    "ACQ-Cooperative Agreement",
    "ACQ-Cooperative Agreement Modification",
    "ACQ-Correspondence",
    "ACQ-Evaluation Worksheet",
    "ACQ-Grant",
    "ACQ-Grant Modification",
    "ACQ-Interagency Agreement",
    "ACQ-Interagency Agreement Status Report",
    "ACQ-Invitation For Bid (IFB)",
    "ACQ-Invoice",
    "ACQ-Notice to Unsuccessful Offeror",
    "ACQ-Pre-award Correspondence",
    "ACQ-Pre-award Correspondence/Protests",
    "ACQ-Presolicitation Document",
    "ACQ-Proposal for Contract, Grant, Agreement",
    "ACQ-Proposal for Modification to Task Order",
    "ACQ-Proposal for Purchase Order",
    "ACQ-Proposal for Task Order",
    "ACQ-Purchase Order",
    "ACQ-Purchase Order Modification",
    "ACQ-Purchase Order Presolicitation Document",
    "ACQ-Request For Proposal (RFP)",
    "ACQ-Request For Quotation (RFQ)",
    "ACQ-Request for Quotation, SF Form18",
    "ACQ-Solicitation Amendment",
    "ACQ-Solicitation, Offer and Award, SF Form 33",
    "ACQ-Solicitation/Contract/Order for Commercial Item, SF FORM 1449",
    "ACQ-Supporting Documentation for Contract, Grant, or Agreement Modification",
    "ACQ-Supporting Documentation for Purchase Order Modification",
    "ACQ-Supporting Documentation for Task Order",
    "ACRS Background Information",
    "ACRS Consultant Report",
    "ACRS Meeting Notebooks",
    "ACRS Most Favored Paragraphs",
    "ACRS Status Report",
    "ACRS Summary Report",
    "ADM Ticket",
    "Administrative Form, GSA",
    "Administrative Form, NRC",
    "Administrative Form, SF",
    "Advance Procurement Plan",
    "Advisory Report",
    "Agreement Request",
    "Agreement States",
    "Agreement States-Regulations Review",
    "Agreement to Transfer Records to the National Archives of the United States, SF Form 258",
    "All Agreement States Letter",
    "Annual Operating Report",
    "Annual Report",
    "Audio File",
    "Audit Plan",
    "Audit Report",
    "Backgrounder",
    "Bankcard Statement and Bankcard Voucher",
    "Biweekly Notice Memoranda",
    "Branch Technical Position",
    "Brochure",
    "Budget Assumptions",
    "Budget Execution Report",
    "Budget Planning and Estimates",
    "Budget Planning Call",
    "Budget, Draft",
    "Budget, Final",
    "Calculation",
    "Capital Planning and Investment Control (CPIC)",
    "Chairman Daily",
    "Charter",
    "Chilling Effect Letter",
    "CNWRA Administrative Procedure",
    "CNWRA Corrective Action Request",
    "CNWRA Instrument Calibration Procedure",
    "CNWRA Instrument Calibration Record",
    "CNWRA Journal Article",
    "CNWRA Procurement Record",
    "CNWRA QA Nonconformance Report",
    "CNWRA QA Surveillance Report",
    "CNWRA Quality Assurance Procedure",
    "CNWRA Reviewer Comments on Journal Article",
    "CNWRA Reviewer Comments on Technical Reports",
    "CNWRA Scientific Notebook and Supplemental Material",
    "CNWRA Scientific Sample Custody Log",
    "CNWRA Software Control Documentation",
    "CNWRA Subcontractor/Consultant Statement of Work",
    "CNWRA Technical Operating Procedure",
    "CNWRA Technical Report",
    "Code of Federal Regulations",
    "Code Relief or Alternative",
    "Codes and Standards",
    "Commercial Contract Solicitation and Award Documents",
    "Commission Action Memoranda (COM)",
    "Commission Closed Meeting Documents",
    "Commission Letter Ballot",
    "Commission Meeting Agenda",
    "Commission Meeting Transcript/Exhibit",
    "Commission Notation Vote",
    "Commission SECY Paper",
    "Commission Staff Requirements Memo (SRM)",
    "Commission Voting Record (CVR)",
    "Committee Letter Report",
    "Communication Plan",
    "Conference Proceeding",
    "Conference/Symposium/Workshop Paper",
    "Confirmatory Action Letter (CAL)",
    "Confirmatory Order",
    "Congressional Affairs Memorandum",
    "Congressional Correspondence",
    "Congressional QAs",
    "Congressional Testimony",
    "Contract Solicitation Docs (RFPs, Invitation to Bid, Request for Qualifications)",
    "Daily Event Report",
    "Database File",
    "Decommissioning Funding Plan DKTs 30, 40, 50, 70",
    "Decommissioning Plan DKTs 30, 40, 50, 70",
    "Deficiency Correspondence (per 10CFR50.55e and Part 21)",
    "Deficiency Report (per 10CFR50.55e and Part 21)",
    "Demand for Information (DFI)",
    "Design Control Document (DCD)",
    "Differing Professional Opinion Case File",
    "Digital Certificate and Broadband User Agreements",
    "Digital Signature/Certificate",
    "DOE Corrective Action Request",
    "DOE YMPO Standard Deficiency Report",
    "Draft Safety Analysis Report (SAR)",
    "Draft Safety Evaluation Report (DSER)",
    "Drawing",
    "E-Mail",
    "EDO Procedure",
    "EDO Procedure Change Notice",
    "Emergency Preparedness-Emergency Plan",
    "Emergency Preparedness-Emergency Plan and Post Exercise Evaluation (FEMA Evaluation)",
    "Emergency Preparedness-Emergency Plan Exercise Objectives and Scenario",
    "Emergency Preparedness-Emergency Plan Implementing Procedures",
    "Emergency Preparedness-EP Position",
    "Emergency Preparedness-FEMA Correspondence to NRC",
    "Emergency Preparedness-NRC Correspondence to FEMA",
    "Emergency Preparedness-Review of Emergency Plan Changes",
    "Emergency Preparedness-Review of Emergency Plan Exercise Objectives and Scenario",
    "Enforcement Action",
    "Enforcement Action Worksheet",
    "Enforcement Guidance Memorandum",
    "Enforcement Manual",
    "Enforcement Manual, Revision",
    "Enforcement Notification",
    "Enforcement Strategy",
    "Enforcement Three Week Memo",
    "Enforcement/Regulatory Conference Invitation/Reply",
    "Enforcement/Regulatory Conference Transcript",
    "Environmental Analysis Statement",
    "Environmental Analysis Statement, Draft",
    "Environmental Assessment",
    "Environmental Impact Appraisal",
    "Environmental Impact Statement",
    "Environmental Monitoring Report",
    "Environmental Protection Plan",
    "Environmental Report",
    "Environmental Report Amendment",
    "Environmental Technical Specification",
    "Equivalent/Clarification, Initial, NRC Form 241",
    "Equivalent/Clarification, NRC Form 241",
    "Equivalent/Clarification, Revision, NRC Form 241",
    "Evacuation Time Estimate/Report (ETE)",
    "Event Report from State",
    "Exemption from NRC Requirements",
    "Exercise of Enforcement Discretion",
    "ExTRA",
    "Facility Safety Evaluation Report",
    "Facsimile",
    "FACT Sheet",
    "Federal Register Notice",
    "Final Safety Analysis Report (FSAR)",
    "Final Safety Evaluation Report (FSER)",
    "Financial Assurance Document",
    "Financial Assurance Package",
    "Finding of No Significant Impact",
    "Fire Protection Plan",
    "FOIA/Privacy Act Background",
    "FOIA/Privacy Act Request",
    "FOIA/Privacy Act Response to Requestor",
    "Foreign Report",
    "Fuel Cycle Reload Report",
    "Fundamental Nuclear Material Control Plan (FNMCP)",
    "General FR Notice Comment Letter",
    "General License Periodic Reports",
    "General Licensee (GL) Registration Form",
    "General Licensee Change Notifications",
    "General Notice (in the Federal Register)",
    "Generic DCD Departures Report",
    "Graphics incl Charts and Tables",
    "Handbook",
    "Highlights",
    "IAEA Safety Guide",
    "Independent Government Cost Estimate - NRC Form 554",
    "Individual Action (Enforcement)",
    "Individual Response to Enforcement Action",
    "INPO Event Report Level 1 (IER 1)",
    "INPO Event Report Level 2 (IER 2)",
    "INPO Event Report Level 3 (IER 3)",
    "INPO Event Report Level 4 (IER 4)",
    "INPO Operations and Maintenance Reminder (OMR)",
    "INPO Significant Event Notification (SEN)",
    "INPO Significant Event Report (SER)",
    "INPO Significant Operating Experience Report (SOER)",
    "INPO Topical Report (TR)",
    "Inservice/Preservice Inspection and Test Report",
    "Inspection Manual",
    "Inspection Manual Change Notice",
    "Inspection Plan",
    "Inspection Report",
    "Inspection Report Correspondence",
    "Inspections, Tests, Analyses, and Acceptance Criteria (ITAAC)",
    "Inspections, Tests, Analyses, and Acceptance Criteria (ITAAC) Closure Notification (ICN)",
    "Integrated Material Performance Evaluation Program (IMPEP)-Agreement States",
    "Integrated Safety Analysis (Plan/Summary/Revision/Update)",
    "Interagency Agreement",
    "International Agreements",
    "International Correspondence, Outgoing",
    "International Nuclear Events Scale (INES) Event Rating Form",
    "Investigative Procedures Manual (IPM)",
    "ITAAC Closure Verification Evaluation Form (VEF)",
    "Journal Article",
    "Legal-Affidavit",
    "Legal-Board Establishment",
    "Legal-Board Notification",
    "Legal-Brief",
    "Legal-Correspondence",
    "Legal-Correspondence/Miscellaneous",
    "Legal-Decision (Partial or Initial)",
    "Legal-Deposition",
    "Legal-Discovery Material",
    "Legal-Discovery Report",
    "Legal-Exhibit",
    "Legal-Final Agency Action Letters and Memoranda",
    "Legal-Finding of Fact/Conclusions of Law",
    "Legal-Hearing File",
    "Legal-Hearing File (For Informal Hearings)",
    "Legal-Hearing Request Referral Memorandum",
    "Legal-Hearing Transcript",
    "Legal-In Camera Filing",
    "Legal-Insurance/Indemnity Document",
    "Legal-Interrogatories and Response",
    "Legal-Intervention Petition, Responses and Contentions",
    "Legal-Limited Appearance Statement",
    "Legal-Memorandum and Order",
    "Legal-Memorandum of Agreement/Understanding",
    "Legal-Motion",
    "Legal-Narrative Testimony",
    "Legal-Notice of Appearance",
    "Legal-Notice of Deposition",
    "Legal-Notice of Hearing",
    "Legal-Notices of Hearing or opportunity for",
    "Legal-Order",
    "Legal-Panel/Board Issuance",
    "Legal-Party Contentions and Associated Pleading",
    "Legal-Petition for Rulemaking",
    "Legal-Petition to Intervene",
    "Legal-Petition To Intervene/Request for Hearing",
    "Legal-Pleading",
    "Legal-Pre-Filed Exhibits",
    "Legal-Pre-Filed Testimony",
    "Legal-Privilege Logs",
    "Legal-Proposed Finding drafted by Parties",
    "Legal-Proposed Finding of Fact and Conclusions of Law",
    "Legal-Public Comment",
    "Legal-Report",
    "Legal-Stipulation/Agreement",
    "Letter",
    "License Fee Requirements Letter",
    "License-Application for (Amend/Renewal/New) for DKT 30, 40, 70",
    "License-Application for Certificate of Compliance (Amend/Renewal/Rev) DKT 71 QA Program",
    "License-Application for Combined License (COLA)",
    "License-Application for Construction Permit DKT 50",
    "License-Application for Design Certification",
    "License-Application for Dry Cask ISFSI DKT 72",
    "License-Application for Early Site Permit (ESP)",
    "License-Application for Export License",
    "License-Application for Facility Operating License (Amend/Renewal) DKT 50",
    "License-Application for HLW Part 63",
    "License-Application for Import License",
    "License-Application for License (Amend/Renewal/New) DKT 40",
    "License-Application for License (Amend/Renewal/New) DKT 70",
    "License-Application for Registry of Sealed Source or Device (Amend/Renewal)",
    "License-Approval for (Amend/Renewal/New) License for DKT 40, 70",
    "License-Approval for MATL Byproduct License (Amend/Renewal/New) DKT 30, 40, 70",
    "License-Certificate of Compliance (Dkt 71)",
    "License-Certificate of Disposition of Materials",
    "License-Combined License (COL)",
    "License-Denial for (Amend/Renewal/New) License for DKT 40, 70",
    "License-Dry Cask, ISFSI, (Amend) DKT 72",
    "License-Early Site Permit (ESP)",
    "License-Exempt Distribution Report",
    "License-Export License Amendment",
    "License-Fee Sheet",
    "License-Fitness for Duty (FFD) Performance Report",
    "License-Fuel Facility Event Evaluation Report",
    "License-General License Notification",
    "License-Import License Amendment",
    "License-Materials Byproduct Amendment DKT 30",
    "License-Monthly Operating Report",
    "License-Negative Declaration of Quality Management Program (QMP) DKT 30",
    "License-No Significant Hazards Consideration Determination and Noticing Action",
    "License-Not elsewhere specified",
    "License-Notification of Authorized Users",
    "License-Operating (New/Renewal/Amendments) DKT 50",
    "License-Operator Examination Report",
    "License-Operator Examination Report (Non-Power Reactors Only)",
    "License-Operator License Exam, Draft",
    "License-Operator, Form 396, Certification of Medical Examination",
    "License-Operator, Form 398, Personal Qualification Statement",
    "License-Operator, Form 474, Simulation Facility Certification",
    "License-Operator, Other HQ and Regional Correspondence",
    "License-Operator, Part 55 Examination Related Material",
    "License-Operator, Report on Interaction (ROI)",
    "License-Operator, Requalification Program Audit",
    "License-QA Program Approval for Radioactive Materials Packages",
    "License-Quality Management Program",
    "License-Registration Certificate for In-Vitro Testing (NRC Form 483)",
    "License-Registry of Sealed Source or Device (New/Amend/Renewal)",
    "License-Renewal, Report on Interaction (ROI)",
    "License-Source Material Amendment DKT 40",
    "License-Special Nuclear Material Amendment DKT 70",
    "License-Technical Assistance Request (TAR)",
    "License-Technical Assistance Request (TAR), Reply to",
    "Licensee 30-Day Written Event Report",
    "Licensee Event Report (LER)",
    "Licensee Performance Review",
    "Licensee Response to Enforcement Action",
    "Licensee Response to Notice of Violation",
    "Limited Work Authorization (LWA) Request",
    "Logbook",
    "Low-Level Waste Manifest Shipping Paper",
    "Management Directive",
    "Manual",
    "Map",
    "Media Briefing Paper",
    "Medical Misadministration Report",
    "Meeting Agenda",
    "Meeting Briefing Package/Handouts",
    "Meeting Minutes",
    "Meeting Notice",
    "Meeting Summary",
    "Meeting Transcript",
    "Memoranda",
    "Monthly SUNSI/SGI Notice Memoranda",
    "Morning Report",
    "MPKI Log",
    "News Article",
    "Newsletter",
    "NMSS Administrative/Management",
    "Non-Agreement States",
    "Non-Cited Violation",
    "Non-Concurrence Process",
    "NON-SES Performance Appraisal System Performance Plan",
    "Note",
    "Note to File incl Telcon Record, Verbal Comm",
    "Notice of Deviation",
    "Notice of Enforcement Discretion (NOED)",
    "Notice of Interagency Meeting",
    "Notice of Non-Conformance",
    "Notice of Return Check",
    "Notice of Violation",
    "Notice of Violation with Proposed Imposition of Civil Penalty",
    "NPDES Noncompliance Notification",
    "NPDES Permit",
    "NRC Administrative Letter",
    "NRC Bulletin",
    "NRC Bulletin, Draft",
    "NRC Circular",
    "NRC Generic Letter",
    "NRC Generic Letter, Draft",
    "NRC Information Notice",
    "NRC Policy Statement",
    "NRC Preliminary Notification of Event/Occurrence",
    "NRC Regulatory Issue Summary",
    "NRO Office Instruction",
    "NRO Safety Evaluation Report (SER)-Delayed",
    "NRR Office Instruction",
    "NRR Office Letter",
    "NUREG",
    "NUREG, Draft",
    "Occupational Exposure Record",
    "OCFO Fee Policy Documentation",
    "OE Annual Report",
    "Official FACA Record for ACRS Meetings",
    "OI Investigation Report",
    "OIG Audit Report",
    "OIG Audit Report Comment",
    "OIG Audit Resolution",
    "OIG Event Inquiry",
    "OMB Clearance Material",
    "OpE Notes and POE",
    "Operating Plan",
    "Operating Procedures",
    "Operating Report",
    "Operating Report, Monthly",
    "OperatingExperience (OpE) Communication",
    "Operational Experience Reports by RES",
    "Order",
    "Order Imposing Civil Monetary Penalty",
    "Order Modifying License",
    "Order Prohibiting",
    "Order Revoking License",
    "Order Suspending License",
    "Order, Confirmatory",
    "Organization Chart",
    "Part 21 Correspondence",
    "Performance Indicator",
    "Performance Plan",
    "Performance Planning and Appraisal (SES)",
    "Periodic Monitoring Report (Radiological/Environmental)",
    "Photograph",
    "Planning Call",
    "Plant Issues Matrix",
    "Plant Performance Review",
    "Plant Status Report",
    "Policy and Program Guidance",
    "Policy Statement",
    "Post-Shutdown Decommissioning Activities Report",
    "Pre-decisional Contract Action",
    "Preliminary Safety Analysis Report (PSAR)",
    "Press Release",
    "Privacy Impact Assessment",
    "Privacy Threshold Analysis",
    "Probabilistic Risk Assessment",
    "Program Review",
    "Project Manager (PM) List",
    "Project Plans and Schedules",
    "Project Requirement Document",
    "Proprietary Information Review",
    "Quality Assurance Program",
    "Radiation Overexposure Reports",
    "Records Retention and Disposal Authorization",
    "Records Transmittal and Receipt, SF Form 135",
    "Reference Safety Analysis Report",
    "Reference Safety Analysis Report, Amendment",
    "Regulatory Analysis",
    "Regulatory Guidance",
    "Regulatory Guide",
    "Regulatory Guide, Draft",
    "Report of Proposed Activities in Non-Agreement States, NRC Form 241",
    "Report, Administrative",
    "Report, Miscellaneous",
    "Report, Technical",
    "Request for Access Authorization",
    "Request for Additional Information (RAI)",
    "Request for OMB Review",
    "Request for Procurement Action (RFPA), NRC Form 400",
    "Request for Review of OMB Reporting Requirements",
    "RES Office Letter",
    "Research Information Letter (RIL)",
    "Resume",
    "Reviewer Comments on Conference/Symposium/Workshop Paper",
    "Route Approval Letter to Licensee",
    "Routine Status Report (Recurring Weekly/Monthly)",
    "Rulemaking- Final Rule",
    "Rulemaking- Proposed Rule",
    "Rulemaking-Authority Statement for EDO Signature",
    "Rulemaking-Comment",
    "Rulemaking-Environmental Assessment",
    "Rulemaking-Environmental Impact Statement",
    "Rulemaking-Plan",
    "Rulemaking-Regulatory Analysis",
    "Rulemaking-Regulatory Plan",
    "Safeguard Incident Report",
    "Safeguards Advisory",
    "Safety and Compliance Inspection Record, NRC Form 591",
    "Safety Evaluation",
    "Safety Evaluation Report",
    "Safety Evaluation Report, Draft",
    "Schedule and Calendars",
    "Security Form-Report of Security Infraction, NRC Form 183",
    "Security Form-Security Incident Report, NRC Form 135",
    "Security Frequently Asked Question (SFAQ)",
    "Security Incidence Report",
    "Security Plan",
    "Security Program",
    "Senior Management Meeting (SMM) Results Letter",
    "Significant Event Report",
    "Site Access Letter",
    "Site Characterization Plan",
    "Site Redress Plan",
    "Site Safety Analysis Report (SSAR)",
    "Slides and Viewgraphs",
    "Social Media-Photograph",
    "Social Media-Video Recording",
    "Software Control Documentation",
    "Software Documentation",
    "Space Management",
    "Space Policy",
    "Special Nuclear Material Physical Inventory Summary Report",
    "Speech",
    "Spreadsheet File",
    "Standard Review Plan",
    "Standard Review Plan Update",
    "Standard Technical Specification incl Change Review Agreement Response",
    "Startup Test Report",
    "State Agreement Application",
    "State Agreement Program Transmittal",
    "Statement of Work",
    "Status Report",
    "Strategic Plan",
    "System Documentation",
    "Task Action Plan",
    "Task Interface Agreement Response (TIA)",
    "Technical Paper",
    "Technical Specification, Amendment",
    "Technical Specification, Bases Change",
    "Technical Specifications",
    "Template",
    "Test/Inspection/Operating Procedure",
    "Text-Safety Report",
    "Threat Advisory",
    "Topical Report",
    "Topical Report Evaluation",
    "Training Evaluation",
    "Training Manual",
    "Transcript",
    "Transportation Route Approval",
    "Trip Report",
    "Updated Final Safety Analysis Report (UFSAR)",
    "User Agreements",
    "Video Recording",
    "Weekly Activities/LEAP (WAR)",
    "Weekly Information Report",
    "Yellow Announcement"
]