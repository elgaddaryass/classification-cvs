import os
import random
import pandas as pd


random.seed(42)

OUTPUT_PATH = "data/cvs.csv"

N_PER_LABEL = 500
# 500 par catégorie = 2500 exemples au total
# Si tu veux 1000 total : mets 200
# Si tu veux 5000 total : mets 1000


DATA = {
    "Software Engineering": {
        "roles": [
            "software engineer", "backend developer", "frontend developer",
            "full stack developer", "web developer", "mobile developer",
            "java developer", "python developer", "php developer",
            "react developer", "qa automation engineer", "software tester",
            "application developer", "junior developer"
        ],
        "skills": [
            "python", "java", "php", "laravel", "django", "flask", "fastapi",
            "html", "css", "javascript", "typescript", "react", "angular",
            "vue", "nodejs", "express", "spring boot", "hibernate",
            "mysql", "postgresql", "mongodb", "sql", "git", "github",
            "gitlab", "rest api", "graphql", "mvc", "oop", "solid",
            "design patterns", "uml", "microservices", "docker",
            "unit testing", "integration testing", "junit", "pytest",
            "selenium", "agile", "scrum", "clean code", "algorithms",
            "data structures", "c", "cpp", "csharp", "dotnet",
            "android", "kotlin", "swift", "flutter", "dart", "firebase"
        ],
        "projects": [
            "developed a web application",
            "built a backend api",
            "created a responsive frontend",
            "designed a mobile application",
            "implemented authentication system",
            "created database models",
            "developed microservices architecture",
            "worked on software testing",
            "built a full stack platform",
            "integrated third party apis"
        ]
    },

    "Data Science": {
        "roles": [
            "data scientist", "data analyst", "machine learning engineer",
            "ai engineer", "business intelligence analyst",
            "data engineer", "nlp engineer", "computer vision engineer",
            "ml engineer", "research assistant in artificial intelligence"
        ],
        "skills": [
            "python", "r", "sql", "pandas", "numpy", "scikit-learn",
            "sklearn", "matplotlib", "seaborn", "statistics",
            "machine learning", "deep learning", "tensorflow", "keras",
            "pytorch", "regression", "classification", "clustering",
            "feature engineering", "model evaluation", "data preprocessing",
            "data cleaning", "data visualization", "power bi", "tableau",
            "excel", "dax", "nlp", "natural language processing",
            "bert", "transformers", "spacy", "nltk", "sentiment analysis",
            "opencv", "yolo", "cnn", "lstm", "time series", "arima",
            "prophet", "forecasting", "spark", "hadoop", "kafka",
            "airflow", "etl", "data pipeline", "mlflow", "kubeflow",
            "model deployment", "big data"
        ],
        "projects": [
            "trained a machine learning model",
            "created a data dashboard",
            "performed exploratory data analysis",
            "built a prediction model",
            "developed a classification system",
            "worked on nlp text classification",
            "created a computer vision model",
            "designed an etl pipeline",
            "cleaned and analyzed large datasets",
            "deployed a machine learning model"
        ]
    },

    "Networks": {
        "roles": [
            "network engineer", "system administrator", "network administrator",
            "noc engineer", "it support technician", "telecom engineer",
            "network technician", "junior network engineer",
            "linux administrator", "windows server administrator"
        ],
        "skills": [
            "network", "networking", "tcp ip", "subnetting", "routing",
            "switching", "cisco", "ccna", "router", "switch", "vlan",
            "lan", "wan", "dns", "dhcp", "ospf", "bgp", "mpls",
            "vpn", "firewall", "wifi", "wireless", "access point",
            "network design", "network infrastructure", "wireshark",
            "snmp", "nagios", "zabbix", "packet analysis",
            "network monitoring", "linux server", "windows server",
            "active directory", "group policy", "powershell",
            "bash", "apache", "nginx", "voip", "sip", "qos",
            "fiber optic", "juniper", "fortinet", "palo alto",
            "ansible", "network automation", "netconf", "restconf"
        ],
        "projects": [
            "configured cisco routers",
            "managed network infrastructure",
            "monitored network traffic",
            "configured dns and dhcp services",
            "deployed linux servers",
            "administered windows server environment",
            "troubleshooted network issues",
            "implemented vlan segmentation",
            "configured vpn access",
            "analyzed packets using wireshark"
        ]
    },

    "Cybersecurity": {
        "roles": [
            "cybersecurity analyst", "soc analyst", "penetration tester",
            "information security analyst", "security engineer",
            "ethical hacker", "incident response analyst",
            "digital forensics analyst", "application security analyst",
            "risk and compliance analyst"
        ],
        "skills": [
            "cybersecurity", "information security", "network security",
            "firewall", "penetration testing", "ethical hacking",
            "kali linux", "metasploit", "burp suite", "nmap",
            "vulnerability assessment", "web security", "owasp",
            "sql injection", "xss", "csrf", "authentication security",
            "siem", "soc", "splunk", "elastic", "log analysis",
            "threat hunting", "incident response", "malware analysis",
            "reverse engineering", "digital forensics", "memory forensics",
            "threat intelligence", "iso 27001", "gdpr", "risk management",
            "security audit", "compliance", "cryptography", "pki",
            "ssl", "tls", "encryption", "endpoint security",
            "ids", "ips", "phishing", "active directory security",
            "red team", "blue team", "devsecops", "sast", "dast"
        ],
        "projects": [
            "performed vulnerability assessment",
            "conducted penetration testing",
            "analyzed security logs",
            "investigated security alerts",
            "secured web applications",
            "implemented firewall rules",
            "worked in security operations center",
            "performed malware analysis",
            "created incident response report",
            "evaluated information security risks"
        ]
    },

    "Cloud DevOps": {
        "roles": [
            "devops engineer", "cloud engineer", "cloud architect",
            "site reliability engineer", "platform engineer",
            "infrastructure engineer", "devsecops engineer",
            "release engineer", "kubernetes administrator",
            "automation engineer"
        ],
        "skills": [
            "devops", "cloud computing", "aws", "azure", "gcp",
            "docker", "kubernetes", "helm", "terraform", "ansible",
            "jenkins", "github actions", "gitlab ci", "ci cd",
            "linux", "bash", "shell scripting", "nginx", "apache",
            "microservices", "deployment", "automation", "monitoring",
            "prometheus", "grafana", "elk", "elasticsearch",
            "logstash", "kibana", "alertmanager", "observability",
            "cloudformation", "pulumi", "infrastructure as code",
            "iac", "aws ec2", "aws s3", "aws lambda", "iam",
            "vpc", "load balancer", "cloudwatch", "azure devops",
            "containerization", "orchestration", "sre",
            "site reliability engineering", "scalability",
            "high availability", "release management"
        ],
        "projects": [
            "deployed applications using docker",
            "managed kubernetes clusters",
            "created ci cd pipelines",
            "automated infrastructure with terraform",
            "configured cloud services",
            "monitored systems with prometheus and grafana",
            "implemented infrastructure as code",
            "managed linux servers",
            "deployed microservices architecture",
            "improved application availability"
        ]
    }
}


COMMON_WORDS = [
    "internship", "junior profile", "engineering student", "computer science",
    "problem solving", "teamwork", "communication", "project management",
    "technical documentation", "agile methodology", "university project",
    "professional experience", "academic project", "certification",
    "training", "self learning", "collaboration"
]


LEVELS = [
    "beginner", "intermediate", "advanced", "strong knowledge of",
    "practical experience with", "academic experience in",
    "hands on experience with", "good understanding of"
]


def create_cv_text(label):
    category_data = DATA[label]

    role = random.choice(category_data["roles"])

    number_of_skills = random.randint(8, 16)
    selected_skills = random.sample(category_data["skills"], number_of_skills)

    number_of_projects = random.randint(2, 4)
    selected_projects = random.sample(category_data["projects"], number_of_projects)

    number_of_common = random.randint(3, 6)
    selected_common = random.sample(COMMON_WORDS, number_of_common)

    level = random.choice(LEVELS)

    text_parts = []

    text_parts.append(role)
    text_parts.append(level)
    text_parts.extend(selected_skills)
    text_parts.extend(selected_projects)
    text_parts.extend(selected_common)

    random.shuffle(text_parts)

    return " ".join(text_parts)


def generate_dataset():
    rows = []

    for label in DATA.keys():
        for _ in range(N_PER_LABEL):
            text = create_cv_text(label)
            rows.append({
                "text": text,
                "label": label
            })

    random.shuffle(rows)

    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Dataset generated successfully: {OUTPUT_PATH}")
    print(f"Total examples: {len(df)}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    generate_dataset()