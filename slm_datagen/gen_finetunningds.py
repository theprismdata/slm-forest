import json
import anthropic
import time
from pathlib import Path
import logging
import glob
import os
import yaml
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config.yaml: {e}")
        return None

# Initialize Anthropic client
config = load_config()
if not config:
    logging.error("Failed to load configuration")
    exit(1)

anthropic_api_key = config.get('langmodel', {}).get('API', {}).get('Claude', {}).get('apikey')
if not anthropic_api_key:
    logging.error("Anthropic API key not found in config.yaml")
    exit(1)

client = anthropic.Anthropic(
    api_key=anthropic_api_key
)

def read_json_data(file_path):
    """Read the source JSON file."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logging.error(f"Expected list but got {type(data)} in {file_path}")
                return None
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

def load_openapi_data():
    """Load and process all OpenAPI data from tax-law-openapi directory."""
    openapi_data = []
    openapi_dir = "tax-law-gen-raw-data/tax-law-openapi"
    
    if not os.path.exists(openapi_dir):
        logging.warning(f"OpenAPI directory not found: {openapi_dir}")
        return openapi_data
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(openapi_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract law information
                    law_info = data.get('법령', {})
                    basic_info = law_info.get('기본정보', {})
                    law_name = basic_info.get('법령명_한글', '')
                    
                    # Process articles
                    articles = law_info.get('조문', {}).get('조문단위', [])
                    for article in articles:
                        article_num = article.get('조문번호', '')
                        article_title = article.get('조문제목', '')
                        article_content = article.get('조문내용', '')
                        
                        # Process items (항)
                        items = article.get('항', [])
                        for item in items:
                            item_num = item.get('항번호', '')
                            item_content = item.get('항내용', '')
                            
                            # Create structured content
                            full_content = f"법령: {law_name}\n조문: {article_num}조"
                            if article_title:
                                full_content += f" ({article_title})"
                            if article_content:
                                full_content += f"\n조문내용: {article_content}"
                            if item_num and item_content:
                                full_content += f"\n항: {item_num} {item_content}"
                            
                            # Create item for processing
                            openapi_item = {
                                'title': f"{law_name} {article_num}조",
                                'content': full_content,
                                'response': f"{law_name} {article_num}조에 따른 규정입니다.",
                                'metadata': {
                                    'source': 'openapi',
                                    'law_name': law_name,
                                    'article_num': article_num,
                                    'item_num': item_num,
                                    'file_path': file_path
                                }
                            }
                            openapi_data.append(openapi_item)
                    
                    logging.info(f"Processed {file_path}: {len(articles)} articles")
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    continue
    
    logging.info(f"Total OpenAPI items loaded: {len(openapi_data)}")
    return openapi_data

def load_judgment_data():
    """Load judgment data from tax-law-judgment directory."""
    judgment_data = []
    judgment_dir = "tax-law-gen-raw-data/tax-law-judgment"
    
    if not os.path.exists(judgment_dir):
        logging.warning(f"Judgment directory not found: {judgment_dir}")
        return judgment_data
    
    # Load main judgment data
    main_file = os.path.join(judgment_dir, "data_main.json")
    if os.path.exists(main_file):
        data = read_json_data(main_file)
        if data:
            for item in data:
                if isinstance(item, dict):
                    # Add source metadata
                    item['metadata'] = item.get('metadata', {})
                    item['metadata']['source'] = 'judgment'
                    judgment_data.append(item)
    
    logging.info(f"Total judgment items loaded: {len(judgment_data)}")
    return judgment_data

def combine_data_sources():
    """Combine data from both OpenAPI and judgment sources."""
    logging.info("Loading OpenAPI data...")
    openapi_data = load_openapi_data()
    
    logging.info("Loading judgment data...")
    judgment_data = load_judgment_data()
    
    # Combine data
    combined_data = openapi_data + judgment_data
    
    logging.info(f"Combined data: {len(combined_data)} total items")
    logging.info(f"  - OpenAPI: {len(openapi_data)} items")
    logging.info(f"  - Judgment: {len(judgment_data)} items")
    
    # Randomly select 1000 items while maintaining the ratio
    total_items = len(combined_data)
    openapi_ratio = len(openapi_data) / total_items
    
    # Calculate how many items to take from each source to maintain ratio
    n_openapi = int(1000 * openapi_ratio)
    n_judgment = 1000 - n_openapi
    
    # Randomly sample from each source
    sampled_openapi = random.sample(openapi_data, n_openapi)
    sampled_judgment = random.sample(judgment_data, n_judgment)
    
    # Combine sampled data
    sampled_data = sampled_openapi + sampled_judgment
    random.shuffle(sampled_data)  # Shuffle the final dataset
    
    logging.info(f"Sampled 1000 items while maintaining source ratio:")
    logging.info(f"  - OpenAPI: {n_openapi} items")
    logging.info(f"  - Judgment: {n_judgment} items")
    
    return sampled_data

def extract_content_parts(content):
    """Safely extract parts from content string."""
    try:
        if not content or not isinstance(content, str):
            return '', '', False
            
        facts = ''
        query = ''
        
        # Extract facts
        if '사실관계' in content:
            parts = content.split('사실관계')
            if len(parts) > 1:
                facts_part = parts[1]
                if '질의요지' in facts_part:
                    facts = facts_part.split('질의요지')[0].strip()
                else:
                    facts = facts_part.strip()
                    
        # Extract query
        if '질의요지' in content:
            parts = content.split('질의요지')
            if len(parts) > 1:
                query_part = parts[1]
                if '관련법령' in query_part:
                    query = query_part.split('관련법령')[0].strip()
                else:
                    query = query_part.strip()
        
        # Check if extraction was successful
        if facts or query:
            return facts, query, True
            
        # Fallback: Try to extract meaningful content without specific markers
        clean_content = content.replace('PDF로 보기', '').replace('상세내용', '').strip()
        paragraphs = [p.strip() for p in clean_content.split('\n') if p.strip()]
        
        if paragraphs:
            # Use first paragraph as facts and second as query if available
            facts = paragraphs[0]
            query = paragraphs[1] if len(paragraphs) > 1 else ''
            return facts, query, False
            
        return '', '', False
        
    except Exception as e:
        logging.error(f"Error extracting content parts: {e}")
        return '', '', False

def generate_distillation_prompt(item):
    """Generate a prompt for knowledge distillation - creating both teacher and student responses."""
    try:
        if not isinstance(item, dict):
            logging.error(f"Invalid item type: {type(item)}")
            return None
            
        content = item.get('content', '')
        title = item.get('title', '')
        response = item.get('response', '')
        metadata = item.get('metadata', {})
        source = metadata.get('source', 'unknown')
        
        # Different prompts for different data sources
        if source == 'openapi':
            # For OpenAPI data (law articles) - create practical user scenarios
            law_name = metadata.get('law_name', '')
            article_num = metadata.get('article_num', '')
            
            prompt = f"""다음 세법 조문을 바탕으로 실제 사용자의 관점에서 자주 물어볼 수 있는 질문과 답변을 생성해주세요.

            법령: {law_name}
            조문번호: {article_num}
            조문내용: {content}

            다음과 같은 형식으로 실용적인 질의응답을 작성해주세요:

            1. 전문가 답변 (상세하고 정확한 설명):
            [질문] 실제 납세자가 물어볼 수 있는 구체적인 상황이나 사례 기반 질문
            [답변] 법적 근거를 포함한 명확하고 상세한 설명, 구체적인 절차나 계산 방법 포함

            2. 일반인을 위한 답변 (쉽게 이해할 수 있는 설명):
            [질문] 같은 내용에 대해 더 일상적인 언어로 표현한 질문
            [답변] 법률 용어를 최소화하고 실생활에 적용할 수 있는 쉬운 설명과 예시 포함"""
            
        else:
            # For judgment data - create practical Q&A based on the case
            facts, query, structured_format = extract_content_parts(content)
            related_laws = metadata.get('related_laws', [])
            
            if structured_format:
                prompt = f"""다음 세금 관련 사례를 바탕으로 실제 납세자의 관점에서 자주 물어볼 수 있는 질문과 답변을 생성해주세요.

                사례 제목: {title}
                사실관계: {facts}
                질의요지: {query}
                판단: {response}
                관련법령: {', '.join(related_laws) if related_laws else ''}

                다음과 같은 형식으로 실용적인 질의응답을 작성해주세요:

                1. 전문가 답변 (구체적인 설명):
                [질문] 이 사례와 관련하여 납세자가 실제로 물어볼 수 있는 구체적인 질문
                [답변] 판례를 근거로 한 명확한 설명, 실무적 적용 방법 포함

                2. 일반인을 위한 답변 (이해하기 쉬운 설명):
                [질문] 같은 내용에 대해 일상적인 언어로 표현한 질문
                [답변] 쉬운 용어로 설명하고 실제 적용 방법 안내"""
            else:
                prompt = f"""다음 세금 관련 내용을 바탕으로 실제 납세자의 관점에서 자주 물어볼 수 있는 질문과 답변을 생성해주세요.

                제목: {title}
                내용: {content}
                결론: {response}
                관련법령: {', '.join(related_laws) if related_laws else ''}

                다음과 같은 형식으로 실용적인 질의응답을 작성해주세요:

                1. 전문가 답변 (구체적인 설명):
                [질문] 이와 관련하여 납세자가 실제로 물어볼 수 있는 구체적인 질문
                [답변] 법적 근거를 포함한 명확한 설명, 실무적 적용 방법 포함

                2. 일반인을 위한 답변 (이해하기 쉬운 설명):
                [질문] 같은 내용에 대해 일상적인 언어로 표현한 질문
                [답변] 쉬운 용어로 설명하고 실제 적용 방법 안내"""
            
        prompt += """
        주의사항:
        1. 질문은 실제 납세자가 물어볼 법한 구체적인 상황이나 사례를 포함해야 합니다
        2. 전문가 답변에는 법적 근거와 정확한 절차를 포함해야 합니다
        3. 일반인 답변은 쉬운 용어를 사용하고 실제 적용 방법을 안내해야 합니다
        4. 모든 내용은 한글로 작성해주세요
        5. 각 답변은 실용적이고 실행 가능한 정보를 제공해야 합니다"""
        
        return prompt
    except Exception as e:
        logging.error(f"Error generating distillation prompt: {e}")
        return None

def parse_distillation_response(response_text):
    """Parse the distillation response to extract teacher and student answers."""
    try:
        if not response_text:
            return None, None
            
        # Split by the expert and general sections
        parts = response_text.split('1. 전문가 답변')
        if len(parts) < 2:
            return None, None
            
        expert_part = parts[1]
        parts = expert_part.split('2. 일반인을 위한 답변')
        if len(parts) < 2:
            return None, None
            
        # Extract expert Q&A
        expert_qa = parts[0].strip()
        expert_parts = expert_qa.split('[질문]')
        if len(expert_parts) > 1:
            expert_q = expert_parts[1].split('[답변]')[0].strip()
            expert_a = expert_parts[1].split('[답변]')[1].strip()
        else:
            return None, None
            
        # Extract student Q&A
        student_qa = parts[1].strip()
        student_parts = student_qa.split('[질문]')
        if len(student_parts) > 1:
            student_q = student_parts[1].split('[답변]')[0].strip()
            student_a = student_parts[1].split('[답변]')[1].strip()
        else:
            return None, None
            
        return (expert_q, expert_a), (student_q, student_a)
        
    except Exception as e:
        logging.error(f"Error parsing distillation response: {e}")
        return None, None

def get_distillation_from_claude(prompt):
    """Get distillation response from Claude API."""
    if not prompt:
        return None
        
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,  # Increased for distillation
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        logging.error(f"Error calling Claude API for distillation: {e}")
        return None

def process_items_with_distillation(json_data, output_file):
    """Process all items and generate QA dataset with distillation."""
    if not json_data:
        logging.error("No data to process")
        return []
        
    qa_pairs = []
    
    # Load existing QA pairs if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            logging.info(f"Loaded {len(qa_pairs)} existing QA pairs")
        except Exception as e:
            logging.error(f"Error loading existing QA pairs: {e}")
            qa_pairs = []
    
    for idx, item in enumerate(json_data):
        logging.info(f"Processing item {idx + 1}/{len(json_data)}")
        
        if not isinstance(item, dict):
            logging.warning(f"Skipping invalid item at index {idx}")
            continue
            
        # Generate distillation prompt
        distillation_prompt = generate_distillation_prompt(item)
        if not distillation_prompt:
            logging.warning(f"Skipping item {idx + 1} due to prompt generation failure")
            continue
            
        # Get distillation response
        distillation_response = get_distillation_from_claude(distillation_prompt)
        
        if distillation_response:
            # Parse distillation response
            expert_qa, student_qa = parse_distillation_response(distillation_response)
            
            if expert_qa and student_qa:
                expert_q, expert_a = expert_qa
                student_q, student_a = student_qa
                
                # Create distillation QA pairs
                teacher_qa = {
                    "question": expert_q,
                    "answer": expert_a,
                    "type": "teacher",
                    "source": item.get('metadata', {}).get('source', 'unknown')
                }
                
                student_qa = {
                    "question": student_q,
                    "answer": student_a,
                    "type": "student",
                    "source": item.get('metadata', {}).get('source', 'unknown')
                }
                
                qa_pairs.extend([teacher_qa, student_qa])
                
                # Save progress periodically
                if (idx + 1) % 10 == 0:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
                        logging.info(f"Saved progress: {len(qa_pairs)} QA pairs")
                    except Exception as e:
                        logging.error(f"Error saving progress: {e}")
                
            else:
                logging.warning(f"Failed to parse distillation response for item {idx + 1}")
                continue
            
        # Rate limiting
        time.sleep(1)
    
    # Final save
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving final output: {e}")
    
    return qa_pairs

def main():
    # Set up paths
    output_file = os.path.join('fine-tunning-ds', "distillation_legal_qa_dataset.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load and combine data from both sources
    logging.info("Starting data combination process...")
    combined_data = combine_data_sources()
    
    if not combined_data:
        logging.error("No data found from any source")
        return
    
    logging.info(f"Total items to process: {len(combined_data)}")
    
    # Process items with distillation
    logging.info("Starting distillation process...")
    distillation_qa_pairs = process_items_with_distillation(combined_data, output_file)
    
    logging.info(f"Generated {len(distillation_qa_pairs)} distillation QA pairs")
    logging.info("Dataset generated successfully!")

if __name__ == "__main__":
    main()