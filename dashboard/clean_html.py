import re

def clean_dashboard_html():
    with open('templates/dashboard.html', 'r') as f:
        content = f.read()
    
    # Verwijder alle regels die beginnen met spaties/tabs gevolgd door nummers en symbolen
    content = re.sub(r'^\s*\d+\s*[→+-].*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+→.*$', '', content, flags=re.MULTILINE)
    
    # Verwijder conversatie markers zoals ⏺
    content = re.sub(r'^⏺.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*⎿.*$', '', content, flags=re.MULTILINE)
    
    # Verwijder regels met alleen nummers
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    # Verwijder regels met ✅ of ☒
    content = re.sub(r'^.*[✅☒].*$', '', content, flags=re.MULTILINE)
    
    # Verwijder "..." regels
    content = re.sub(r'^\s*\.\.\.\s*$', '', content, flags=re.MULTILINE)
    
    # Verwijder extra lege regels
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Verwijder conversatie content tussen > en ⏺
    content = re.sub(r'^>.*$', '', content, flags=re.MULTILINE)
    
    # Schrijf clean versie
    with open('templates/dashboard_cleaned.html', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    clean_dashboard_html()
    print("Dashboard HTML cleaned!")