#comment transformer une prédiction simple en analyse intelligente.



from sklearn.metrics.pairwise import cosine_similarity


class CVAgent:
    def __init__(self, model):
        """
        L'agent reçoit le modèle entraîné.
        Ce modèle sert à prédire la catégorie des CVs.
        """
        self.model = model

    def analyze_single_cv(self, cv_text):
        """
        Analyse un seul CV :
        - prédire la catégorie
        - extraire les compétences
        """

        category = self.model.predict([cv_text])[0]
        skills = self.extract_skills(cv_text)

        return {
            "category": category,
            "skills": skills
        }

    def rank_cvs(self, profile_text, cvs):
        """
        Cette fonction reçoit :
        - profile_text : le profil recherché écrit par l'utilisateur
        - cvs : une liste de CVs, chaque CV contient name et text

        Elle retourne :
        - les CVs classés du meilleur au moins bon selon le score final
        """

        results = []

        # 1. Prédire la catégorie du profil recherché
        profile_category = self.model.predict([profile_text])[0]

        # 2. Extraire les compétences demandées dans le profil
        profile_skills = self.extract_skills(profile_text)

        # 3. Analyser chaque CV envoyé
        for cv in cvs:
            cv_name = cv["name"]
            cv_text = cv["text"]

            # Prédire la catégorie du CV
            cv_category = self.model.predict([cv_text])[0]

            # Extraire les compétences du CV
            cv_skills = self.extract_skills(cv_text)

            # Calculer la similarité entre le profil recherché et le CV
            similarity_score = self.compute_similarity(profile_text, cv_text)

            # Calculer le score des compétences communes
            skills_score = self.compute_skills_score(profile_skills, cv_skills)

            # Vérifier si la catégorie du CV correspond à la catégorie du profil
            if profile_category == cv_category:
                category_score = 1
            else:
                category_score = 0

            # Score final sur 100
            final_score = (
                similarity_score * 40
                + skills_score * 50
                + category_score * 10
            )

            # Transformer le score en priorité
            priority = self.get_priority(final_score)

            results.append({
                "cv_name": cv_name,
                "profile_category": profile_category,
                "cv_category": cv_category,
                "profile_skills": profile_skills,
                "cv_skills": cv_skills,
                "similarity_score": round(similarity_score * 100, 2),
                "skills_score": round(skills_score * 100, 2),
                "final_score": round(final_score, 2),
                "priority": priority
            })

        # 4. Classer les CVs du meilleur score au plus faible
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        return results

    def compute_similarity(self, profile_text, cv_text):
        """
        Calcule la similarité entre le profil recherché et le texte du CV.
        On utilise le TF-IDF déjà entraîné dans le modèle.
        """

        tfidf = self.model.named_steps["tfidf"]

        vectors = tfidf.transform([profile_text, cv_text])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        return similarity

    def compute_skills_score(self, profile_skills, cv_skills):
        """
        Calcule le pourcentage des compétences demandées qui existent dans le CV.
        """

        if len(profile_skills) == 0:
            return 0

        profile_skills_set = set(profile_skills)
        cv_skills_set = set(cv_skills)

        common_skills = profile_skills_set.intersection(cv_skills_set)

        return len(common_skills) / len(profile_skills_set)
    def get_priority(self, score):
      if score >= 50:
        return "Priorité élevée"

      if score >= 30:
        return "Priorité moyenne"

      return "Priorité faible"

    def extract_skills(self, text):
        """
        Cherche les compétences techniques présentes dans le texte.
        """

        skills_list = [
            # Développement logiciel
            "python",
            "java",
            "php",
            "laravel",
            "sql",
            "mysql",
            "postgresql",
            "mongodb",
            "html",
            "css",
            "javascript",
            "typescript",
            "react",
            "angular",
            "vue",
            "nodejs",
            "express",
            "django",
            "flask",
            "fastapi",
            "spring boot",
            "hibernate",
            "git",
            "github",
            "gitlab",
            "docker",
            "kubernetes",
            "oop",
            "object oriented programming",
            "uml",
            "api",
            "rest api",
            "microservices",
            "mvc",
            "solid",
            "design patterns",
            "testing",
            "unit testing",
            "integration testing",
            "selenium",
            "junit",
            "pytest",
            "agile",
            "scrum",

            # Data science / IA
            "machine learning",
            "deep learning",
            "data analysis",
            "data science",
            "pandas",
            "numpy",
            "scikit-learn",
            "sklearn",
            "tensorflow",
            "keras",
            "pytorch",
            "nlp",
            "natural language processing",
            "bert",
            "transformers",
            "spacy",
            "nltk",
            "opencv",
            "yolo",
            "cnn",
            "lstm",
            "power bi",
            "tableau",
            "excel",
            "statistics",
            "regression",
            "classification",
            "clustering",
            "big data",
            "spark",
            "hadoop",
            "kafka",
            "airflow",
            "etl",
            "mlflow",

            # Réseaux
            "network",
            "networking",
            "cisco",
            "routing",
            "switching",
            "tcp ip",
            "dns",
            "dhcp",
            "linux",
            "windows server",
            "active directory",
            "firewall",
            "vpn",
            "lan",
            "wan",
            "vlan",
            "ospf",
            "bgp",
            "wireshark",
            "snmp",
            "nagios",
            "zabbix",
            "juniper",
            "fortinet",
            "palo alto",
            "ccna",

            # Cybersécurité
            "cybersecurity",
            "security",
            "network security",
            "penetration testing",
            "ethical hacking",
            "kali linux",
            "metasploit",
            "burp suite",
            "nmap",
            "siem",
            "soc",
            "splunk",
            "elastic",
            "owasp",
            "xss",
            "sql injection",
            "csrf",
            "malware analysis",
            "reverse engineering",
            "forensics",
            "cryptography",
            "encryption",
            "iso 27001",
            "gdpr",
            "risk management",
            "devsecops",

            # Cloud / DevOps
            "aws",
            "azure",
            "gcp",
            "cloud computing",
            "terraform",
            "ansible",
            "jenkins",
            "github actions",
            "gitlab ci",
            "ci cd",
            "prometheus",
            "grafana",
            "nginx",
            "apache",
            "helm",
            "sre",
            "cloudformation",
            "infrastructure as code",
            "monitoring",
            "deployment",
            "automation"
        ]

        text = text.lower()

        detected_skills = []

        for skill in skills_list:
            if skill in text:
                detected_skills.append(skill)

        return detected_skills