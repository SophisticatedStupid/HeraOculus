import { runOneWayAnova, ptukey, qtukey } from './src/services/statistics/anova';

// Test: clearly unequal variances with larger sample sizes to trigger Games-Howell
console.log('Test: Unequal variances (should trigger Games-Howell)');
const r1 = runOneWayAnova({ 
  response: [
    // Group A: small variance
    10, 11, 12, 10, 11, 12, 10, 11,
    // Group B: large variance  
    1, 20, 2, 25, 3, 30, 4, 35, 5, 40,
    // Group C: small variance
    50, 51, 52, 50, 51, 52, 50, 51
  ],
  group: ['A','A','A','A','A','A','A','A',
          'B','B','B','B','B','B','B','B','B','B',
          'C','C','C','C','C','C','C','C']
});
console.log('equalVar:', r1.assumptions.equalVariancesAssumed);
console.log('posthoc methods:', r1.posthoc.map(p => p.method));
console.log('posthoc df values:', r1.posthoc.map(p => p.df));
r1.posthoc.forEach((p, i) => {
  console.log(`Pair ${i+1}: ${p.pair.join(' vs ')} | method=${p.method} | p=${p.p.toFixed(4)} | df=${p.df}`);
});

// Test: Verify k=2 exact ptukey against known values
console.log('\nTest: ptukey k=2 exact values');
// For k=2, q=3.926 (df=4) should give p≈0.05 (the critical value at alpha=0.05)
const p1 = ptukey(3.926486, 2, 4);
console.log('ptukey(3.926486, 2, 4) =', p1.toFixed(6), '(expected ~0.05)');

// qtukey(0.95, 2, 4) should give ~3.926 (the critical value)
const q1 = qtukey(0.95, 2, 4);
console.log('qtukey(0.95, 2, 4) =', q1.toFixed(4), '(expected ~3.9265)');

// For large df, qtukey(0.95, 2, 100) ≈ 2.8070
const q2 = qtukey(0.95, 2, 100);
console.log('qtukey(0.95, 2, 100) =', q2.toFixed(4), '(expected ~2.8070)');

console.log('\nAll tests completed successfully!');