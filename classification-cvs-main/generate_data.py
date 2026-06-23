import pandas as pd

data = [
    # Software Engineering
    ("python java laravel sql git html css javascript", "Software Engineering"),
    ("php laravel mysql web development backend frontend", "Software Engineering"),
    ("react angular nodejs typescript frontend backend fullstack", "Software Engineering"),
    ("c++ object oriented programming design patterns algorithms", "Software Engineering"),
    ("django flask python web framework postgresql rest api", "Software Engineering"),
    ("java spring hibernate maven junit testing ci cd", "Software Engineering"),
    ("javascript typescript nodejs express mongodb restful api", "Software Engineering"),
    ("ruby rails mvc postgresql heroku git agile scrum", "Software Engineering"),
    ("android kotlin java mobile development firebase google play", "Software Engineering"),
    ("ios swift xcode mobile app development cocoapods", "Software Engineering"),

    # Data Science
    ("machine learning python pandas numpy statistics scikit-learn", "Data Science"),
    ("deep learning tensorflow keras pytorch neural network", "Data Science"),
    ("data analysis excel power bi python visualization matplotlib", "Data Science"),
    ("nlp natural language processing spacy bert transformers", "Data Science"),
    ("data engineering spark hadoop airflow etl pipeline sql", "Data Science"),
    ("r programming statistics regression classification clustering", "Data Science"),
    ("big data hadoop hive mapreduce hdfs spark streaming kafka", "Data Science"),
    ("mlops mlflow kubeflow model deployment monitoring", "Data Science"),
    ("time series forecasting arima lstm prophet anomaly detection", "Data Science"),
    ("opencv image processing object detection yolo cnn", "Data Science"),

    # Networks
    ("cisco routing switching tcp ip network security ccna", "Networks"),
    ("linux server windows server tcp ip dns dhcp", "Networks"),
    ("network administration firewall vpn switching routing ospf", "Networks"),
    ("wireless wifi 802.11 network design infrastructure lan wan", "Networks"),
    ("network monitoring nagios wireshark packet analysis snmp", "Networks"),
    ("sdwan network automation ansible python netconf devnet", "Networks"),
    ("ipv6 mpls qos voip sip network engineering telecom", "Networks"),
    ("juniper aruba fortinet paloalto network security", "Networks"),

    # Cybersecurity
    ("cybersecurity firewall penetration testing siem soc", "Cybersecurity"),
    ("security audit vulnerability kali linux network security", "Cybersecurity"),
    ("iso 27001 gdpr compliance risk management information security", "Cybersecurity"),
    ("malware analysis reverse engineering forensics threat intelligence", "Cybersecurity"),
    ("web application security owasp burp suite sql injection xss", "Cybersecurity"),
    ("red team blue team ctf security operations center", "Cybersecurity"),
    ("cryptography pki ssl tls certificates encryption", "Cybersecurity"),
    ("splunk elastic siem log analysis threat hunting detection", "Cybersecurity"),

    # Cloud DevOps
    ("docker kubernetes aws linux ci cd deployment terraform", "Cloud DevOps"),
    ("cloud computing docker gitlab ci linux automation devops", "Cloud DevOps"),
    ("aws azure gcp cloud architect solutions architect", "Cloud DevOps"),
    ("jenkins github actions ci cd pipeline automation testing", "Cloud DevOps"),
    ("terraform infrastructure as code iac pulumi cloudformation", "Cloud DevOps"),
    ("prometheus grafana monitoring observability alerting elk", "Cloud DevOps"),
    ("kubernetes helm operators deployment microservices orchestration", "Cloud DevOps"),
    ("site reliability engineering sre chaos engineering scalability", "Cloud DevOps"),
]

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/cvs.csv", index=False)
print(f"✅ Dataset enrichi : {len(df)} exemples")
print(df["label"].value_counts())