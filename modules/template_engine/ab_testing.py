# modules/template_engine/ab_testing.py

import json
import logging
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from engine.storage import SecureStorage
from ai.nlp import NLPProcessor


class TestStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TestMetric(Enum):
    OPEN_RATE = "open_rate"
    CLICK_RATE = "click_rate"
    REPLY_RATE = "reply_rate"
    CONVERSION_RATE = "conversion_rate"
    UNSUBSCRIBE_RATE = "unsubscribe_rate"
    SPAM_RATE = "spam_rate"
    DELIVERABILITY_RATE = "deliverability_rate"


@dataclass
class TestVariant:
    id: str
    test_id: str
    name: str
    template_id: str
    template_content: str
    subject_line: str
    weight: float = 0.5  # Traffic distribution weight
    is_control: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTest:
    id: str
    name: str
    description: str
    campaign_id: str
    status: TestStatus
    primary_metric: TestMetric
    secondary_metrics: List[TestMetric] = field(default_factory=list)
    variants: List[TestVariant] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class TestResult:
    variant_id: str
    metric: TestMetric
    value: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float
    is_winner: bool = False
    improvement: Optional[float] = None


class ABTestingEngine:
    def __init__(self, SecureStorage: SecureStorage, nlp_processor: NLPProcessor):
        self.SecureStorage = SecureStorage
        self.nlp = nlp_processor
        self.logger = logging.getLogger("ab_testing_engine")
        self.logger.setLevel(logging.INFO)
        
        # Set up logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize database tables
        self._initialize_tables()
        
        # Load active tests
        self.active_tests: Dict[str, ABTest] = {}
        self._load_active_tests()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS ab_tests (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            campaign_id TEXT NOT NULL,
            status TEXT NOT NULL,
            primary_metric TEXT NOT NULL,
            secondary_metrics TEXT,
            start_date TEXT,
            end_date TEXT,
            min_sample_size INTEGER DEFAULT 1000,
            confidence_level REAL DEFAULT 0.95,
            created_at TEXT,
            created_by TEXT,
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS test_variants (
            id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            name TEXT NOT NULL,
            template_id TEXT NOT NULL,
            template_content TEXT NOT NULL,
            subject_line TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            is_control INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY (test_id) REFERENCES ab_tests (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS test_events (
            id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            lead_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (test_id) REFERENCES ab_tests (id),
            FOREIGN KEY (variant_id) REFERENCES test_variants (id),
            FOREIGN KEY (lead_id) REFERENCES leads (id)
        )
        """)

        self.SecureStorage.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            sample_size INTEGER NOT NULL,
            confidence_interval TEXT,
            p_value REAL,
            is_winner INTEGER DEFAULT 0,
            improvement REAL,
            calculated_at TEXT,
            FOREIGN KEY (test_id) REFERENCES ab_tests (id),
            FOREIGN KEY (variant_id) REFERENCES test_variants (id)
        )
        """)

    def _load_active_tests(self):
        """Load active tests from SecureStorage"""
        for row in self.SecureStorage.query("SELECT * FROM ab_tests WHERE status = ?", (TestStatus.RUNNING.value,)):
            test = ABTest(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                campaign_id=row['campaign_id'],
                status=TestStatus(row['status']),
                primary_metric=TestMetric(row['primary_metric']),
                secondary_metrics=json.loads(row['secondary_metrics']) if row['secondary_metrics'] else [],
                start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else None,
                end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                min_sample_size=row['min_sample_size'],
                confidence_level=row['confidence_level'],
                created_at=datetime.fromisoformat(row['created_at']),
                created_by=row['created_by']
            )
            
            # Load variants
            test.variants = []
            for v_row in self.SecureStorage.query("SELECT * FROM test_variants WHERE test_id = ?", (test.id,)):
                variant = TestVariant(
                    id=v_row['id'],
                    test_id=v_row['test_id'],
                    name=v_row['name'],
                    template_id=v_row['template_id'],
                    template_content=v_row['template_content'],
                    subject_line=v_row['subject_line'],
                    weight=v_row['weight'],
                    is_control=bool(v_row['is_control']),
                    created_at=datetime.fromisoformat(v_row['created_at'])
                )
                test.variants.append(variant)
            
            self.active_tests[test.id] = test

    def create_test(self, name: str, description: str, campaign_id: str,
                   primary_metric: TestMetric, secondary_metrics: List[TestMetric] = None,
                   min_sample_size: int = 1000, confidence_level: float = 0.95,
                   created_by: str = "system") -> ABTest:
        """Create a new A/B test"""
        test_id = f"test_{int(datetime.now().timestamp())}"
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            campaign_id=campaign_id,
            status=TestStatus.DRAFT,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            created_by=created_by
        )
        
        # Save test to database
        self.SecureStorage.execute(
            """
            INSERT INTO ab_tests 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test.id,
                test.name,
                test.description,
                test.campaign_id,
                test.status.value,
                test.primary_metric.value,
                json.dumps([m.value for m in test.secondary_metrics]),
                test.start_date.isoformat() if test.start_date else None,
                test.end_date.isoformat() if test.end_date else None,
                test.min_sample_size,
                test.confidence_level,
                test.created_at.isoformat(),
                test.created_by
            )
        )
        
        self.logger.info(f"Created A/B test: {test.name} ({test.id})")
        return test

    def add_variant(self, test_id: str, name: str, template_id: str,
                   template_content: str, subject_line: str,
                   weight: float = 0.5, is_control: bool = False) -> TestVariant:
        """Add a variant to an A/B test"""
        variant_id = f"variant_{int(datetime.now().timestamp())}"
        
        variant = TestVariant(
            id=variant_id,
            test_id=test_id,
            name=name,
            template_id=template_id,
            template_content=template_content,
            subject_line=subject_line,
            weight=weight,
            is_control=is_control
        )
        
        # Save variant to database
        self.SecureStorage.execute(
            """
            INSERT INTO test_variants 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                variant.id,
                variant.test_id,
                variant.name,
                variant.template_id,
                variant.template_content,
                variant.subject_line,
                variant.weight,
                1 if variant.is_control else 0,
                variant.created_at.isoformat()
            )
        )
        
        # Update test in memory if active
        if test_id in self.active_tests:
            self.active_tests[test_id].variants.append(variant)
        
        self.logger.info(f"Added variant to test {test_id}: {variant.name}")
        return variant

    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        # Get test from database
        test = self._get_test(test_id)
        if not test:
            self.logger.error(f"Test not found: {test_id}")
            return False
        
        # Check if test has at least 2 variants
        if len(test.variants) < 2:
            self.logger.error(f"Test {test_id} must have at least 2 variants")
            return False
        
        # Update test status
        test.status = TestStatus.RUNNING
        test.start_date = datetime.now()
        
        self.SecureStorage.execute(
            """
            UPDATE ab_tests 
            SET status = ?, start_date = ? 
            WHERE id = ?
            """,
            (test.status.value, test.start_date.isoformat(), test_id)
        )
        
        # Add to active tests
        self.active_tests[test_id] = test
        
        self.logger.info(f"Started A/B test: {test.name} ({test_id})")
        return True

    def pause_test(self, test_id: str) -> bool:
        """Pause an A/B test"""
        if test_id not in self.active_tests:
            self.logger.error(f"Active test not found: {test_id}")
            return False
        
        test = self.active_tests[test_id]
        test.status = TestStatus.PAUSED
        
        self.SecureStorage.execute(
            "UPDATE ab_tests SET status = ? WHERE id = ?",
            (test.status.value, test_id)
        )
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        self.logger.info(f"Paused A/B test: {test.name} ({test_id})")
        return True

    def complete_test(self, test_id: str) -> bool:
        """Complete an A/B test"""
        if test_id not in self.active_tests:
            self.logger.error(f"Active test not found: {test_id}")
            return False
        
        test = self.active_tests[test_id]
        test.status = TestStatus.COMPLETED
        test.end_date = datetime.now()
        
        self.SecureStorage.execute(
            """
            UPDATE ab_tests 
            SET status = ?, end_date = ? 
            WHERE id = ?
            """,
            (test.status.value, test.end_date.isoformat(), test_id)
        )
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        # Calculate final results
        self._calculate_test_results(test_id)
        
        self.logger.info(f"Completed A/B test: {test.name} ({test_id})")
        return True

    def cancel_test(self, test_id: str) -> bool:
        """Cancel an A/B test"""
        if test_id not in self.active_tests:
            self.logger.error(f"Active test not found: {test_id}")
            return False
        
        test = self.active_tests[test_id]
        test.status = TestStatus.CANCELLED
        test.end_date = datetime.now()
        
        self.SecureStorage.execute(
            """
            UPDATE ab_tests 
            SET status = ?, end_date = ? 
            WHERE id = ?
            """,
            (test.status.value, test.end_date.isoformat(), test_id)
        )
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        self.logger.info(f"Cancelled A/B test: {test.name} ({test_id})")
        return True

    def _get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test from database"""
        row = self.SecureStorage.query("SELECT * FROM ab_tests WHERE id = ?", (test_id,)).fetchone()
        if not row:
            return None
        
        test = ABTest(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            campaign_id=row['campaign_id'],
            status=TestStatus(row['status']),
            primary_metric=TestMetric(row['primary_metric']),
            secondary_metrics=json.loads(row['secondary_metrics']) if row['secondary_metrics'] else [],
            start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else None,
            end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
            min_sample_size=row['min_sample_size'],
            confidence_level=row['confidence_level'],
            created_at=datetime.fromisoformat(row['created_at']),
            created_by=row['created_by']
        )
        
        # Load variants
        test.variants = []
        for v_row in self.SecureStorage.query("SELECT * FROM test_variants WHERE test_id = ?", (test_id,)):
            variant = TestVariant(
                id=v_row['id'],
                test_id=v_row['test_id'],
                name=v_row['name'],
                template_id=v_row['template_id'],
                template_content=v_row['template_content'],
                subject_line=v_row['subject_line'],
                weight=v_row['weight'],
                is_control=bool(v_row['is_control']),
                created_at=datetime.fromisoformat(v_row['created_at'])
            )
            test.variants.append(variant)
        
        return test

    def get_variant_for_lead(self, lead_id: str, test_id: str) -> Optional[TestVariant]:
        """Get the variant to show to a lead for a test"""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Check if lead has already been assigned to a variant
        existing_event = self.SecureStorage.query(
            """
            SELECT variant_id FROM test_events 
            WHERE test_id = ? AND lead_id = ? 
            ORDER BY timestamp DESC LIMIT 1
            """,
            (test_id, lead_id)
        ).fetchone()
        
        if existing_event:
            # Return the same variant for consistency
            for variant in test.variants:
                if variant.id == existing_event['variant_id']:
                    return variant
        
        # Assign a new variant based on weights
        return self._assign_variant_by_weight(test.variants)

    def _assign_variant_by_weight(self, variants: List[TestVariant]) -> TestVariant:
        """Assign a variant based on weights"""
        # Normalize weights
        total_weight = sum(v.weight for v in variants)
        if total_weight == 0:
            # Equal distribution if no weights
            return random.choice(variants)
        
        normalized_weights = [v.weight / total_weight for v in variants]
        
        # Weighted random selection
        return random.choices(variants, weights=normalized_weights)[0]

    def track_event(self, test_id: str, variant_id: str, lead_id: str,
                   event_type: str, metadata: Dict = None):
        """Track an event for a test variant"""
        event_id = f"event_{int(datetime.now().timestamp())}"
        
        self.SecureStorage.execute(
            """
            INSERT INTO test_events 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                test_id,
                variant_id,
                lead_id,
                event_type,
                json.dumps(metadata) if metadata else None,
                datetime.now().isoformat()
            )
        )

    def _calculate_test_results(self, test_id: str):
        """Calculate results for a completed test"""
        test = self._get_test(test_id)
        if not test:
            return
        
        # Get all variants
        variants = test.variants
        
        # Calculate results for each metric
        all_metrics = [test.primary_metric] + test.secondary_metrics
        
        for metric in all_metrics:
            # Get event counts for each variant
            variant_data = {}
            
            for variant in variants:
                # Get total sent (leads assigned to this variant)
                sent = self.SecureStorage.query(
                    """
                    SELECT COUNT(DISTINCT lead_id) as count 
                    FROM test_events 
                    WHERE test_id = ? AND variant_id = ?
                    """,
                    (test_id, variant.id)
                ).fetchone()[0]
                
                # Get event count for this metric
                event_count = 0
                if metric == TestMetric.OPEN_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'open'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.CLICK_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'click'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.REPLY_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'reply'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.CONVERSION_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'conversion'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.UNSUBSCRIBE_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'unsubscribe'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.SPAM_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'spam'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                elif metric == TestMetric.DELIVERABILITY_RATE:
                    event_count = self.SecureStorage.query(
                        """
                        SELECT COUNT(DISTINCT lead_id) as count 
                        FROM test_events 
                        WHERE test_id = ? AND variant_id = ? AND event_type = 'delivered'
                        """,
                        (test_id, variant.id)
                    ).fetchone()[0]
                
                # Calculate rate
                rate = event_count / sent if sent > 0 else 0
                
                variant_data[variant.id] = {
                    'sent': sent,
                    'events': event_count,
                    'rate': rate
                }
            
            # Perform statistical analysis
            if len(variants) >= 2:
                # Get control variant
                control_variant = next((v for v in variants if v.is_control), None)
                if not control_variant:
                    control_variant = variants[0]
                
                control_data = variant_data[control_variant.id]
                
                for variant in variants:
                    if variant.id == control_variant.id:
                        continue
                    
                    variant_data_item = variant_data[variant.id]
                    
                    # Calculate p-value using chi-square test for proportions
                    contingency_table = [
                        [control_data['events'], control_data['sent'] - control_data['events']],
                        [variant_data_item['events'], variant_data_item['sent'] - variant_data_item['events']]
                    ]
                    
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    
                    # Calculate confidence interval
                    rate_diff = variant_data_item['rate'] - control_data['rate']
                    se = stats.sem([control_data['rate'], variant_data_item['rate']])
                    ci = stats.norm.interval(0.95, loc=rate_diff, scale=se)
                    
                    # Calculate improvement
                    improvement = (variant_data_item['rate'] - control_data['rate']) / control_data['rate'] if control_data['rate'] > 0 else 0
                    
                    # Determine winner
                    is_winner = p_value < (1 - test.confidence_level) and rate_diff > 0
                    
                    # Save result
                    result_id = f"result_{test_id}_{variant.id}_{metric.value}"
                    self.SecureStorage.execute(
                        """
                        INSERT OR REPLACE INTO test_results 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result_id,
                            test_id,
                            variant.id,
                            metric.value,
                            variant_data_item['rate'],
                            variant_data_item['sent'],
                            json.dumps(ci),
                            p_value,
                            1 if is_winner else 0,
                            improvement,
                            datetime.now().isoformat()
                        )
                    )

    def get_test_results(self, test_id: str) -> Dict:
        """Get results for a test"""
        results = {}
        
        # Get test info
        test = self._get_test(test_id)
        if not test:
            return results
        
        # Get all metrics to analyze
        all_metrics = [test.primary_metric] + test.secondary_metrics
        
        for metric in all_metrics:
            metric_results = []
            
            # Get results for each variant
            for row in self.SecureStorage.query(
                """
                SELECT v.*, r.value, r.sample_size, r.confidence_interval, 
                       r.p_value, r.is_winner, r.improvement
                FROM test_variants v
                LEFT JOIN test_results r ON v.id = r.variant_id AND r.metric = ?
                WHERE v.test_id = ?
                """,
                (metric.value, test_id)
            ):
                result = {
                    'variant_id': row['id'],
                    'variant_name': row['name'],
                    'is_control': bool(row['is_control']),
                    'value': row['value'],
                    'sample_size': row['sample_size'],
                    'confidence_interval': json.loads(row['confidence_interval']) if row['confidence_interval'] else None,
                    'p_value': row['p_value'],
                    'is_winner': bool(row['is_winner']),
                    'improvement': row['improvement']
                }
                metric_results.append(result)
            
            results[metric.value] = metric_results
        
        return results

    def get_test_summary(self, test_id: str) -> Dict:
        """Get a summary of test results"""
        results = self.get_test_results(test_id)
        if not results:
            return {}
        
        # Get test info
        test = self._get_test(test_id)
        if not test:
            return {}
        
        # Find winner for primary metric
        primary_results = results.get(test.primary_metric.value, [])
        winner = None
        if primary_results:
            winner = next((r for r in primary_results if r['is_winner']), None)
        
        # Calculate total sample size
        total_sample_size = sum(r['sample_size'] for r in primary_results)
        
        # Calculate test duration
        duration = None
        if test.start_date and test.end_date:
            duration = (test.end_date - test.start_date).days
        
        return {
            'test_id': test_id,
            'test_name': test.name,
            'status': test.status.value,
            'primary_metric': test.primary_metric.value,
            'total_sample_size': total_sample_size,
            'duration_days': duration,
            'winner': winner,
            'all_metrics': list(results.keys())
        }

    def get_recommendations(self, test_id: str) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Get test results
        results = self.get_test_results(test_id)
        if not results:
            return recommendations
        
        # Get test info
        test = self._get_test(test_id)
        if not test:
            return recommendations
        
        # Analyze primary metric
        primary_results = results.get(test.primary_metric.value, [])
        if not primary_results:
            return recommendations
        
        # Find winner and runner-up
        sorted_results = sorted(primary_results, key=lambda x: x['value'], reverse=True)
        winner = sorted_results[0]
        runner_up = sorted_results[1] if len(sorted_results) > 1 else None
        
        # Generate recommendations
        if winner['is_winner']:
            recommendations.append(
                f"Variant '{winner['variant_name']}' is the winner for {test.primary_metric.value} "
                f"with a {winner['improvement']:.1%} improvement over control."
            )
            
            if winner['improvement'] > 0.1:  # 10% improvement
                recommendations.append(
                    f"Consider implementing '{winner['variant_name']}' as the new default template."
                )
        
        if runner_up:
            if runner_up['p_value'] > 0.05:
                recommendations.append(
                    f"The difference between variants is not statistically significant "
                    f"(p={runner_up['p_value']:.3f}). Consider running the test longer."
                )
        
        # Check sample size
        total_sample_size = sum(r['sample_size'] for r in primary_results)
        if total_sample_size < test.min_sample_size:
            recommendations.append(
                f"Sample size ({total_sample_size}) is below the recommended minimum ({test.min_sample_size}). "
                f"Consider running the test longer to get more reliable results."
            )
        
        # Check for statistical significance
        if not any(r['is_winner'] for r in primary_results):
            recommendations.append(
                f"No variant showed statistically significant improvement. "
                f"Consider testing different variables or a larger sample size."
            )
        
        return recommendations

    def export_test_data(self, test_id: str, format: str = "csv") -> str:
        """Export test data in specified format"""
        # Get test results
        results = self.get_test_results(test_id)
        if not results:
            return ""
        
        # Get test info
        test = self._get_test(test_id)
        if not test:
            return ""
        
        # Prepare data for export
        export_data = []
        
        for metric_name, metric_results in results.items():
            for result in metric_results:
                export_data.append({
                    'test_id': test_id,
                    'test_name': test.name,
                    'metric': metric_name,
                    'variant_id': result['variant_id'],
                    'variant_name': result['variant_name'],
                    'is_control': result['is_control'],
                    'value': result['value'],
                    'sample_size': result['sample_size'],
                    'confidence_interval_low': result['confidence_interval'][0] if result['confidence_interval'] else None,
                    'confidence_interval_high': result['confidence_interval'][1] if result['confidence_interval'] else None,
                    'p_value': result['p_value'],
                    'is_winner': result['is_winner'],
                    'improvement': result['improvement']
                })
        
        if format.lower() == "csv":
            df = pd.DataFrame(export_data)
            return df.to_csv(index=False)
        elif format.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_ab_test_stats(self, days: int = 30) -> Dict:
        """Get A/B testing statistics"""
        since = datetime.now() - timedelta(days=days)
        
        # Get test counts by status
        status_counts = {status.value: 0 for status in TestStatus}
        total_tests = 0
        
        for row in self.SecureStorage.query(
            "SELECT status, COUNT(*) as count FROM ab_tests WHERE created_at >= ? GROUP BY status",
            (since.isoformat(),)
        ):
            status_counts[row['status']] = row['count']
            total_tests += row['count']
        
        # Get average improvement for winning tests
        avg_improvement = self.SecureStorage.query(
            """
            SELECT AVG(improvement) as avg_imp 
            FROM test_results r
            JOIN ab_tests t ON r.test_id = t.id
            WHERE r.is_winner = 1 AND t.created_at >= ?
            """,
            (since.isoformat(),)
        ).fetchone()[0] or 0
        
        # Get most tested metrics
        metric_counts = {}
        for row in self.SecureStorage.query(
            """
            SELECT primary_metric, COUNT(*) as count 
            FROM ab_tests 
            WHERE created_at >= ? 
            GROUP BY primary_metric 
            ORDER BY count DESC
            """,
            (since.isoformat(),)
        ):
            metric_counts[row['primary_metric']] = row['count']
        
        return {
            'total_tests': total_tests,
            'status_distribution': status_counts,
            'average_improvement': avg_improvement,
            'most_tested_metrics': metric_counts
        }
