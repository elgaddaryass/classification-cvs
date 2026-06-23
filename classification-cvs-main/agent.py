#comment transformer une prédiction simple en analyse intelligente.





class CVAgent:
    def __init__(self, model):
        self.model = model

    def analyze(self, text):
        category = self.model.predict([text])[0]
        skills = self.extract_skills(text)
        recommendation = self.recommend(category)

        return {
            "category": category,
            "skills": skills,
            "recommendation": recommendation
        }

    def extract_skills(self, text):
        skills_list = [
            # Programming languages
            "python", "java", "c", "c++", "c#", "php", "javascript", "typescript",
            "ruby", "kotlin", "swift", "dart", "r", "scala", "go",

            # Web development
            "html", "css", "react", "angular", "vue", "nodejs", "express",
            "django", "flask", "fastapi", "laravel", "spring boot", "hibernate",
            "bootstrap", "tailwind", "rest api", "graphql", "wordpress",

            # Databases
            "sql", "mysql", "postgresql", "mongodb", "oracle", "sql server",
            "redis", "firebase", "snowflake",

            # Software engineering
            "oop", "object oriented programming", "design patterns", "uml",
            "solid", "clean architecture", "microservices", "mvc", "testing",
            "unit testing", "integration testing", "selenium", "junit", "pytest",
            "git", "github", "gitlab", "agile", "scrum",

            # Data science / AI
            "machine learning", "deep learning", "data analysis", "data science",
            "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras",
            "pytorch", "matplotlib", "seaborn", "power bi", "tableau", "excel",
            "statistics", "regression", "classification", "clustering",
            "nlp", "bert", "transformers", "spacy", "nltk", "opencv", "yolo",
            "cnn", "lstm", "arima", "prophet", "mlflow", "kubeflow",

            # Networks
            "network", "networking", "cisco", "routing", "switching", "tcp ip",
            "dns", "dhcp", "ospf", "bgp", "vlan", "lan", "wan", "vpn",
            "wireshark", "snmp", "nagios", "zabbix", "juniper", "fortinet",
            "palo alto", "linux server", "windows server", "active directory",
            "ccna", "mpls", "voip", "sip", "qos",

            # Cybersecurity
            "cybersecurity", "security", "firewall", "penetration testing",
            "ethical hacking", "kali linux", "metasploit", "burp suite",
            "nmap", "owasp", "sql injection", "xss", "csrf", "siem", "soc",
            "splunk", "elastic", "threat hunting", "malware analysis",
            "reverse engineering", "forensics", "iso 27001", "gdpr",
            "risk management", "cryptography", "pki", "ssl", "tls",
            "encryption", "devsecops",

            # Cloud / DevOps
            "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
            "ansible", "jenkins", "github actions", "gitlab ci", "ci cd",
            "cloud computing", "cloud architect", "cloudformation", "pulumi",
            "prometheus", "grafana", "elk", "helm", "sre", "site reliability engineering",
            "nginx", "apache", "bash", "powershell"
]

        
        
        
        
        
        
        
        text = text.lower()

        return [skill for skill in skills_list if skill in text]

    def recommend(self, category):
        if category == "Software Engineering":
            return "This profile is suitable for software development internships."

        if category == "Data Science":
            return "This profile is suitable for data science internships."

        if category == "Networks":
            return "This profile is suitable for network and system administration internships."

        if category == "Cybersecurity":
            return "This profile is suitable for cybersecurity internships."

        if category == "Cloud DevOps":
            return "This profile is suitable for cloud and DevOps internships."

        return "This profile needs more analysis."